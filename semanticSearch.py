import numpy as np
import streamlit as st

name_to_image = {
            "flower": "images/flower.jpg",
            "car": "images/car.jpg",
            "tree": "images/tree.jpg",
            "mountain": "images/mountain.jpg",
            "building":"images/building.jpg"
        }
words = name_to_image.keys()

class Embeddings:
    def __init__(self, embedding_dimension=25):
        """
        Initialize the class
        """
        # Your code here
        print("Initializing the embedding dict ...")
        self.embedding_dict = self.load_glove_embeddings(embedding_dimension)
        print("Done!")

    def load_glove_embeddings(self, embedding_dimension):
        file_location = f"../glove_twitter/glove.twitter.27B.{embedding_dimension}d.txt"
        embedding_dict = {}
        with open(file_location, encoding='utf-8') as f:
            for line in f:
                word = line.split(" ")[0]
                embedding = np.array([float(val) for val in line.split(" ")[1:]])
                embedding_dict[word] = embedding 
        return embedding_dict

    def get_embedding(self, word):
        if word in self.embedding_dict:
            return self.embedding_dict[word]
        return self.embedding_dict['<unknown>']


def find_N_nearest_words(embed, input, N):
  input_embedding = embed.get_embedding(input)
  similarities=[]
  for w in words:
    w_embedding = embed.get_embedding(w)
    similarity = np.dot(input_embedding, w_embedding)
    similarities.append([similarity,w])
  similarities.sort(reverse=True)
  return [w for s,w in similarities[:N]]


@st.cache_resource
def load_embed():
    embed = Embeddings(100)
    return embed

def main():
    st.title("Find The Nearest Image App")
    prev_input=''
    user_input = st.text_input("Enter a word:")
    if prev_input != user_input:
        prev_input = user_input
        nearest_matched_word = find_N_nearest_words(load_embed(), user_input, 1)[0]
        st.caption("Find the nearest image: "+nearest_matched_word)
        st.image(name_to_image.get(nearest_matched_word))
    

if __name__ == "__main__":
  main()