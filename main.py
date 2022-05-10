
import pandas as pd
import streamlit as st
from keras import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
#
def main():
    MODEL_PATH = 'model/saved_model.pb'

    st.title("Mesh on Demand")

    text_example = st.selectbox("Example of the texts:",
                            ("Type headache moreover people type headache time treatment basis diagnosis technically term content diagnostic process base classification disorder beta produce classification base distinction headache application uniform concept come treatment type headache",
                            "BACKGROUND: Indonesia is the largest island country in the world with diverse ethnicity and cultural backgrounds. This study aimed to understand the variation in attitudes toward epilepsy among the Javanese, Sundanese, and the Minahasa ethnic groups in Indonesia. METHOD: This study recruited Sundanese from Tasikmalaya and Minahasan from Manado using the Indonesian Public Attitudes Toward Epilepsy (PATE) scale. The results were compared to the Javanese and Malaysian data in previous studies. RESULT: A total of 200 respondents, 100 from each ethnic group were recruited, with a mean age of 38.51\u202fyears. They were predominantly females (54%) and had secondary education level or lower (56.67%). The Javanese had a higher total mean score, indicating poorer attitudes toward epilepsy, as compared to the Minahasan and Sundanese groups. These differences were noted in the personal domain, but not the general domain. There were no significant differences in the mean scores in both personal and general domains between the Minahasan, Sundanese, and Malaysian populations. Subanalysis on the aspects of life showed that the Javanese had a significantly higher score in the aspects of education, marital relationship, and employment. CONCLUSION: The attitudes toward epilepsy were similar between the Indonesian (Sundanese and the Minahasan) and Malaysian, except the Javanese with poorer attitude. These differences could be socioeconomically or culturally related.")
                            )


    keras_model = models.load_model('model')

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 30000
    # Max number of words in each abstract.
    MAX_SEQUENCE_LENGTH = 250
    # This is fixed.
    EMBEDDING_DIM = 300

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

    X_one = []
    X_one.append(
        "Type headache moreover people type headache time treatment basis diagnosis technically term content diagnostic process base classification disorder beta produce classification base distinction headache application uniform concept come treatment type headache")
    X_one_only = np.array(X_one)
    X_one_only = tokenizer.texts_to_sequences(X_one_only)
    X_one_only = pad_sequences(X_one_only, maxlen=MAX_SEQUENCE_LENGTH)

    preds = keras_model.predict(X_one_only)

    tags = pd.get_dummies(pd.Series(y).apply(pd.Series).stack()).sum(level=0)
    tags.columns.values




if __name__ == "__main__":
    main()