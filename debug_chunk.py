import pickle

with open("vector_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

for i, c in enumerate(chunks):
    if "and expose the Company to significant licensing" in c["text"]:
        print("MATCH AT INDEX", i)
        print("--- previous chunk ---")
        print(chunks[i - 1]["text"])
        print("--- this chunk ---")
        print(c["text"])
        break