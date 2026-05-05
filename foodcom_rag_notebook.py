# Databricks notebook source
# MAGIC %md
# MAGIC # Food.com Recipe Recommender — RAG Chatbot
# MAGIC **ISA 632 Group Project | Miami University**
# MAGIC
# MAGIC Builds a RAG chatbot recommending recipes from Food.com using
# MAGIC Databricks Vector Search and Llama 3. Follows Labs 2, 3, and 4 patterns.
# MAGIC
# MAGIC **Use Classic Compute — not Serverless.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Install Dependencies

# COMMAND ----------

# MAGIC %pip install -qqq -U databricks-sdk databricks-langchain databricks-vectorsearch langchain==0.3.27 mlflow-skinny[databricks]>=3.4.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Configuration

# COMMAND ----------

import time
import pandas as pd
from pprint import pprint
from databricks.vector_search.client import VectorSearchClient

CATALOG   = "isa632_7474656346303369"
SCHEMA    = "greenjc7"
YOUR_MUID = "greenjc7"

TABLE_NAME       = f"{CATALOG}.{SCHEMA}.recipes_clean"
VS_ENDPOINT_NAME = f"vs_endpoint_{YOUR_MUID}"
VS_INDEX_NAME    = f"{CATALOG}.{SCHEMA}.recipe_embeddings"
MODEL_NAME       = f"{CATALOG}.{SCHEMA}.foodcom_recipe_recommender"
LLM_ENDPOINT     = "databricks-meta-llama-3-3-70b-instruct"

RAW_RECIPES_PATH      = "/Volumes/isa632_7474656346303369/default/foodrecipes/RAW_recipes.csv"
RAW_INTERACTIONS_PATH = "/Volumes/isa632_7474656346303369/default/foodrecipes/RAW_interactions.csv"

SAMPLE_SIZE = 5000  # Set to None for full 231K after testing

print(f"Catalog      : {CATALOG}")
print(f"Schema       : {SCHEMA}")
print(f"Table        : {TABLE_NAME}")
print(f"VS Endpoint  : {VS_ENDPOINT_NAME}")
print(f"VS Index     : {VS_INDEX_NAME}")
print(f"Sample size  : {SAMPLE_SIZE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Load & Preprocess Data

# COMMAND ----------

print("Loading RAW_recipes.csv ...")
recipes_pd = pd.read_csv(RAW_RECIPES_PATH)
print(f"  Shape: {recipes_pd.shape}")

print("Loading RAW_interactions.csv ...")
interactions_pd = pd.read_csv(RAW_INTERACTIONS_PATH)
print(f"  Shape: {interactions_pd.shape}")

if SAMPLE_SIZE:
    recipes_pd = recipes_pd.sample(n=min(SAMPLE_SIZE, len(recipes_pd)), random_state=42)
    print(f"\nDev sample: {len(recipes_pd):,} recipes")

# Compute average rating per recipe from interactions
avg_ratings = (
    interactions_pd
    .groupby("recipe_id")["rating"]
    .mean()
    .round(2)
    .reset_index()
    .rename(columns={"rating": "avg_rating"})
)

# Merge — RAW_recipes uses 'id', interactions uses 'recipe_id'
recipes_pd = (
    recipes_pd
    .merge(avg_ratings, left_on="id", right_on="recipe_id", how="left")
    .rename(columns={"id": "recipe_id"})
)

# Clean
recipes_pd["minutes"]     = recipes_pd["minutes"].clip(upper=1440)
recipes_pd["description"] = recipes_pd["description"].fillna("")
recipes_pd["avg_rating"]  = recipes_pd["avg_rating"].fillna(0.0).round(2)
recipes_pd["name"]        = recipes_pd["name"].fillna("Unnamed Recipe")

# COMMAND ----------

meal_keywords = ['breakfast,', 'breakfast.', 'breakfast']
def is_target_meal(tags_val):
    if isinstance(tags_val, list):
        return any(kw in tag.lower() for tag in tags_val for kw in meal_keywords)
    if isinstance(tags_val, str):
        return any(kw in tags_val.lower() for kw in meal_keywords)
    return False

recipes_pd = recipes_pd[recipes_pd["tags"].apply(is_target_meal)].reset_index(drop=True)
print(f"Breakfast recipes: {len(recipes_pd):,}")

# COMMAND ----------

# Build embed_text — mirrors Lab 2's CONCAT of useful text fields
def make_embed_text(row):
    return " | ".join([
        str(row.get("name", "")),
        str(row.get("description", "")),
        "Ingredients: " + str(row.get("ingredients", "")),
        "Tags: "        + str(row.get("tags", "")),
    ])

recipes_pd["embed_text"] = recipes_pd.apply(make_embed_text, axis=1)

keep = [
    "recipe_id", "name", "minutes", "n_steps", "steps",
    "description", "ingredients", "tags", "nutrition",
    "n_ingredients", "avg_rating", "embed_text"
]
recipes_pd = recipes_pd[[c for c in keep if c in recipes_pd.columns]]

print(f"\nFinal shape: {recipes_pd.shape}")
print(f"Sample embed_text:\n{recipes_pd['embed_text'].iloc[0][:250]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Write Delta Table
# MAGIC Writes to your existing schema — no catalog or schema creation needed.

# COMMAND ----------

recipes_spark = spark.createDataFrame(recipes_pd)

(
    recipes_spark
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

# Required by Vector Search (Lab 2 pattern)
spark.sql(f"""
    ALTER TABLE {TABLE_NAME}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

count = spark.table(TABLE_NAME).count()
print(f"Table: {TABLE_NAME} ({count:,} rows)")
print("Change Data Feed: enabled")

display(spark.table(TABLE_NAME).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Create Vector Search Index
# MAGIC Uses helper functions from `Classroom-Setup-Common.py`.
# MAGIC Your endpoint `vs_endpoint_greenjc7` must exist first.
# MAGIC If not: Databricks UI > Compute > Vector Search > Create endpoint.

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_full_name):
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get('status').get('ready', False)
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e) and 'NOT_FOUND' not in str(e):
            raise e
    return False

def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
    for i in range(180):
        idx = vsc.get_index(vs_endpoint_name, index_name).describe()
        index_status = idx.get('status', idx.get('index_status', {}))
        status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
        if "ONLINE" in status:
            print(f"Index '{index_name}' is ready.")
            return
        if "UNKNOWN" in status:
            print(f"Status unknown — assuming ready.")
            return
        elif "PROVISIONING" in status:
            if i % 40 == 0:
                print(f"Waiting for index to be ready ... {index_status}")
            time.sleep(10)
        else:
            raise Exception(f"Error with index: {idx}")
    raise Exception("Timeout — index not ready.")

vsc = VectorSearchClient(disable_notice=True)

if not index_exists(vsc, VS_ENDPOINT_NAME, VS_INDEX_NAME):
    print(f"Creating index '{VS_INDEX_NAME}' ...")
    vsc.create_delta_sync_index(
        endpoint_name                 = VS_ENDPOINT_NAME,
        index_name                    = VS_INDEX_NAME,
        source_table_name             = TABLE_NAME,
        pipeline_type                 = "TRIGGERED",
        primary_key                   = "recipe_id",
        embedding_source_column       = "embed_text",
        embedding_model_endpoint_name = "databricks-gte-large-en",
    )
else:
    print("Index exists — syncing ...")
    vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME).sync()

wait_for_index_to_be_ready(vsc, VS_ENDPOINT_NAME, VS_INDEX_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Test Vector Search
# MAGIC Same `similarity_search` test as Lab 2.

# COMMAND ----------

index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)

question = "quick chicken dinner with lemon and garlic"

results = index.similarity_search(
    query_text  = question,
    columns     = ["recipe_id", "name", "minutes", "avg_rating"],
    num_results = 4
)

docs = results.get("result", {}).get("data_array", [])
print(f"Query: {question}\n")
pprint(docs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Set Environment Variables & Enable MLflow Tracing
# MAGIC Lab 2 pattern — pass VS index name to agent.py via env var.

# COMMAND ----------

import os
import mlflow

os.environ["VS_INDEX_NAME"]     = VS_INDEX_NAME
os.environ["LLM_ENDPOINT_NAME"] = LLM_ENDPOINT

mlflow.langchain.autolog()

print(f"VS_INDEX_NAME     = {os.environ['VS_INDEX_NAME']}")
print(f"LLM_ENDPOINT_NAME = {os.environ['LLM_ENDPOINT_NAME']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — Load Agent and Test
# MAGIC `agent.py` must be in the same Workspace folder as this notebook.

# COMMAND ----------

from agent import AGENT

user_input = "What is a good quick pasta recipe for a weeknight dinner?"

request = {
    "input": [
        {"role": "user", "content": user_input}
    ]
}

resp = AGENT.predict(request)
print(resp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 — Register Model to Unity Catalog
# MAGIC Lab 2 pattern: Models-from-Code via `python_model="agent.py"`.

# COMMAND ----------

from mlflow.models.resources import DatabricksVectorSearchIndex
from pkg_resources import get_distribution

mlflow.set_registry_uri("databricks-uc")
mlflow.models.set_model(AGENT)

with mlflow.start_run(run_name="foodcom_rag_register") as run:
    model_info = mlflow.pyfunc.log_model(
        name         = "agent",
        python_model = "agent.py",
        pip_requirements=[
            f"langchain=={get_distribution('langchain').version}",
            f"databricks-vectorsearch=={get_distribution('databricks-vectorsearch').version}",
            f"databricks_langchain=={get_distribution('databricks_langchain').version}",
            f"mlflow=={get_distribution('mlflow').version}",
        ],
        resources=[
            DatabricksVectorSearchIndex(index_name=VS_INDEX_NAME)
        ],
    )

model_uri     = f"runs:/{run.info.run_id}/{model_info.name}"
model_version = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Registered: {MODEL_NAME}  version: {model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10 — Evaluate with MLflow
# MAGIC Lab 3 pattern: `predict_fn` + built-in scorers + custom Guidelines scorer.

# COMMAND ----------

import mlflow.pyfunc
from mlflow.genai import evaluate
from mlflow.genai.scorers import Guidelines, RelevanceToQuery, Safety

mlflow.set_registry_uri("databricks-uc")

eval_model_uri = f"models:/{MODEL_NAME}/{model_version.version}"
model = mlflow.pyfunc.load_model(eval_model_uri)

@mlflow.trace
def predict_fn(messages):
    out = model.predict({"input": messages})
    if isinstance(out, dict) and "output" in out:
        try:
            return out["output"][-1]["content"][0]["text"].strip()
        except Exception:
            pass
    if isinstance(out, str):
        return out.strip()
    return str(out).strip()

eval_data = pd.DataFrame({
    "inputs": [
        {"input": [{"role": "user", "content": "Easy chocolate cake for beginners?"}]},
        {"input": [{"role": "user", "content": "I have chicken, garlic, lemon. Quick dinner?"}]},
        {"input": [{"role": "user", "content": "High protein meal prep for the week?"}]},
        {"input": [{"role": "user", "content": "Easy weeknight dinner under 30 minutes?"}]},
        {"input": [{"role": "user", "content": "Popular Mexican dishes on Food.com?"}]},
    ],
    "expected_response": [
        "A chocolate cake recipe with beginner-friendly instructions.",
        "A recipe using chicken, garlic, and lemon with short cook time.",
        "A high protein recipe such as chicken or beef.",
        "A recipe that takes 30 minutes or less.",
        "A Mexican recipe such as tacos or enchiladas.",
    ]
})

grounded_scorer = Guidelines(
    name="recipe_grounded",
    guidelines="""The response must recommend at least one specific recipe by name.
It must include an estimated cook time and why it fits the request.
It must NOT invent recipes — only use retrieved results.
Keep within 400 words."""
)

mlflow.set_experiment(f"/Users/{YOUR_MUID}/FoodCom_RAG_Eval")

with mlflow.start_run(run_name="foodcom_rag_eval"):
    eval_results = evaluate(
        data       = eval_data,
        predict_fn = predict_fn,
        scorers    = [grounded_scorer, RelevanceToQuery(), Safety()]
    )

display(eval_results.metrics)
display(eval_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11 — Deploy with Model Serving
# MAGIC Lab 4 pattern: `agents.deploy()` + wait loop + print Review App URL.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks import agents

clean_model = MODEL_NAME.replace(" ", "")
clean_index = VS_INDEX_NAME.replace(" ", "")

deployment_info = agents.deploy(
    clean_model,
    model_version    = model_version.version,
    scale_to_zero    = True,
    environment_vars = {
        "VS_INDEX_NAME":     clean_index,
        "LLM_ENDPOINT_NAME": LLM_ENDPOINT,
    }
)

w = WorkspaceClient()
print("Waiting for endpoint — this takes 15–20 minutes.", end="")

while (
    w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY
    or
    w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS
):
    print(".", end="")
    time.sleep(30)

print("\nEndpoint is ready!")
print(f"Endpoint URL   : {deployment_info.endpoint_url}")
print(f"Review App URL : {deployment_info.review_app_url}")