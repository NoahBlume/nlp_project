# nlp_project
final project for cs 4650/7650

This project was created to analyze headlines from MSNBC and Fox News regarding COVID-19. We set out to see if there was a difference in how the two organizations covered the pandemic using natural language processing techniques.

Directory Structure:
- All python code is located in the top level directory.
- Sample headlines are contained in nlp_project/headlines
- Examples of the intermidiate and final stages of our data analysis are contained in nlp_project/analyzed_data

Installation Instructions:
- You may need to install the following libraries with pip or any other preferred methods:
  - stanza
    - note: stanza requires that you have pytorch installed
  - textblob
  - sklearn
  - yellowbrick
  - numpy
  
 Running the code (I used Python 3.7 while developing the code):
 - run "python run_all.py" from the top-level directory. 
 - This will process all of the articles, compute their tfidf vectors, and group them via k means. 
 - The whole process will take about 25 minutes. If you just wish to look at the final output of the process, see the files in nlp_project/analyzed_data
