import pickle

print("Loading old vector database...")

with open("vectorstore/data.pkl", "rb") as file:
    data = pickle.load(file)

documents = data["documents"]

print(f"Found {len(documents)} documents.")

with open("vectorstore/data_light.pkl", "wb") as file:
    pickle.dump(documents, file)

print("✅ data_light.pkl created successfully.")