"""Small experiment: does the embedding model understand meaning?"""

from sentence_transformers import SentenceTransformer, util

# Downloads the model (~440 MB) the first time, then it is cached.
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

question = "Trials for NSCLC that exclude patients with brain metastases"

cards = [
    "Non-small cell lung carcinoma. Patients with untreated CNS metastases are not eligible.",
    "Breast cancer trial. Patients with brain metastases may enroll.",
    "Non-small cell lung cancer study of osimertinib; primary outcome is overall survival.",
    "A recipe for chocolate cake with vanilla frosting.",
    "Non-small cell lung cancer trial. Patients with brain metastases are eligible.",
]

question_vector = model.encode(question, normalize_embeddings=True)
card_vectors = model.encode(cards, normalize_embeddings=True)

print(f"Each text becomes {len(question_vector)} numbers.")
print(f"First 5 numbers of the question: {question_vector[:5]}\n")

scores = util.cos_sim(question_vector, card_vectors)[0]

for score, card in sorted(zip(scores.tolist(), cards), reverse=True):
    print(f"{score:.3f}  {card}")