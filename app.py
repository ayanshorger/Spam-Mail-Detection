import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load dataset
data = pd.read_csv(r"C:\Users\Ayan Shorger\OneDrive\Desktop\SpamMailDetection\spam.csv")

#remove the duplicates data from the data set
data.drop_duplicates(inplace=True)


#replacing the name ham = Not Spam and spam = Spam (camecase)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])


# Prepare training data
message = data['Message']
category = data['Category']

message_train, message_test, category_train, category_test = train_test_split(message, category, test_size=0.2)

#convert text data set into numerical data set using contvectorizer to train owr model
vectorizer = CountVectorizer(stop_words='english') #terminate comman english words like 'a'
X_train_vectorized = vectorizer.fit_transform(message_train) #convert data set intp numerical data set


#Creating model
model = MultinomialNB()
model.fit(X_train_vectorized, category_train) #traing the model waya converting data set

# Prediction function
def predict(message):
    input_vector = vectorizer.transform([message])
    prediction = model.predict(input_vector)
    return prediction[0]

# --- Streamlit UI ---
st.set_page_config(page_title="Spam Mail Detector", layout="centered")

st.title("📧 Spam Mail Detector")
st.write("Enter a message below and find out if it's **Spam** or **Not Spam**.")

user_input = st.text_area("✉️ Enter your message here:")

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        result = predict(user_input)
        if result == "Spam":
            st.error("🚫 This message is **SPAM**.")
        else:
            st.success("✅ This message is **NOT SPAM**.")
