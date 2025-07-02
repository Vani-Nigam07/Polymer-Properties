import os

print("🔁 Running Ingestion Pipeline...")
os.system("python src/pipeline/ingestion_pipeline.py")

print("\n🔁 Running User Behavior Analysis Pipeline...")
os.system("python src/pipeline/process_conversations.py")

print("\n🧠 Running NBA Engine Pipeline...")
os.system("python src/pipeline/nba_pipeline.py")

print("\n✅ All pipelines executed successfully.")
