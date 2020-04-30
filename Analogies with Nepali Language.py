#!/usr/bin/env python
# coding: utf-8

# ##  Analogies with Nepali Language
# 

# In[1]:


import numpy as np


# In[2]:


## File Paths
word_2_vec = "nepali_embeddings_word2vec.txt"


# In[3]:


def read_word_to_vecs(pretrained_file):
    with open(pretrained_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map


# In[4]:


words, word_to_vec_map = read_word_to_vecs(word_2_vec)


# In[5]:


def cosine_similarity(u,v): 
    """
    u : Word vector
    V : Word vector
    
    Returns : 
        Cosine similarity
    
    """
    # dot product of the word vectors
    dot_product = np.dot(u,v)
    # L2 norms ie. Distance
    dist_u = np.sqrt(np.sum(u**2))
    dist_v = np.sqrt(np.sum(v**2))
    cosine_similarity = dot_product/(dist_u*dist_v)
    return cosine_similarity


# In[7]:


word_u = word_to_vec_map["ठमेल"]
word_v = word_to_vec_map["न्यूरोड"]
cosine_similarity(word_u,word_v)


# In[18]:


def analogy_finder(word_a, word_b, word_c, word_2_vec_map, word_list):
    e_a, e_b, e_c = word_2_vec_map[word_a],word_to_vec_map[word_b],word_to_vec_map[word_c]
    words = word_list
    max_cosine_similarity = -100.00 # Big negative number
    final_word = None 
    input_words_set = set([word_a, word_b, word_c])
    for word in words:
        # If same words are found
        if word in input_words_set : 
            continue
            
        # Otherwise, compute the cosine similarity
        sim = cosine_similarity((e_b-e_a), (word_2_vec_map[word]-e_c))
        
        if sim > max_cosine_similarity : 
            print(word)
            max_cosine_similarity = sim
            final_word = word
        
    return final_word


# In[19]:


word_a = "केटा"
word_b = "केटी"
word_c = "राजा"
analogy_finder(word_a,word_b,word_c,word_to_vec_map, words)


# In[ ]:





# In[ ]:




