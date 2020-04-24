# nlp_project
final project for cs 4650/7650

This project was created to analyze headlines from MSNBC and Fox News regarding COVID-19. We set out to see if there was a difference in how the two organizations covered the pandemic using natural language processing techniques.

Directory Structure:
- All python code is located in the top level directory.
- Sample headlines are contained in /headlines
- Examples of the intermidiate and final stages of our data analysis are contained in /analyzed_data

Installation Instructions:
- You may need to install the following libraries with pip or any other preffered methods:
  - stanza
  - textblob
  - sklearn
  - yellowbrick
  
 Running the code:
 - Get an API key from newsapi.org and paste it into the "YOUR API KEY HERE" part of headline_retriever.py
 - If you do not have an API key, I have uploaded some sample headline files in the /headlines directory. You can load these with the load_articles function in headline_retriever.
 - Once you have obtained the headlines, use pre_preocessor.py to tokenize and lemmatize the headlines, and do sentiment analysis on them. Pass the loaded headlines into the pre_process function to do this.
 - Use tfidf_vectorizer.py to create the tfidf vector for each preprocessed headline. Do this by first passing the preprocessed headlines as a list into the fit_vectorizer function. Then pass the same headlines into the add_tfidf_vectors function.
 - To run kmeans on the articles using the tfidf vectors and sentiment scores use kmeans_clusterer.py. Pass the tfidf-vectorized articles into the run_kmeans_and_label_articles function. This will label each headline with a cluster group for each of various runs of k means.
- If you wish to organize the articles so that all headlines in one group are contained together in a list, pass the kmeans-labelled articles into the create_cluster_groups function in kmeans_cluster.py.
- To save any of these article lists to a json file, pass the articles into the save_articles function in headline_retriever.py.

Optional commands:
- If you wish to only include headlines that contain words related to COVID-19 you can pass any of the article lists in to the filter function in the filter.py file. It is recommended that you apply this before adding the tfidf vectors.
