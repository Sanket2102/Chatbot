import numpy as np
import pickle
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


## Downloading wordnet from nltk
import nltk
# nltk.download("wordnet")

## Downloading stopwords from nltk
nltk.download("stopwords")

#Loading the pickle files

# tokenizer for one hot encoding
with open('./pickles/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# label encoder object for class classification
with open("./pickles/label_encoder.pkl","rb") as file:
    encoder = pickle.load(file)

# creating label dictionary

labels = dict(zip(encoder.classes_, range(len(encoder.classes_))))
# interchanging key value pairs
labels = {value: key for key, value in labels.items()}

# deep learning model
with open("./pickles/model.pkl","rb") as file:
    model = pickle.load(file)


responses = {'cancel_order':["To cancel your order, go to your profile, select 'Your Orders', then choose the product you want to cancel. Tap the three dots in the top right and select 'Cancel Order'. If you still face any issue contact us on someone@example.com."],
            'change_order': ["To remove or add items in your order, you can use the update order option. Go to your profile, select 'Your Orders', then tap the three dots in the top right and select 'Update Order'. Please note that you can not add items in your order once the order is dispatched. For more information contact us on someone@example.com"], 
            'change_shipping_address': ["To change shipping address, go to your profile, select 'Saved Address', then click on 'add new address'. To make this your default address for future orders, tap on the three dots next to the address and select 'Set as Default'. If you still face any issue contact us on someone@example.com and our support team will reach out to you as soon as possible."],
            'check_cancellation_fee': ["You will not be charged any cancellation fee for the orders that are cancelled before dispatch. However, if the order is already dispatched you will be charged a minimal fee of 50Rs per item. For more help, write to us on someone@example.com"], 
            'check_invoice': ["You can download the invoice from your purchase in the 'Your Orders' section. Go to your profile, select 'Your Orders', tap the three dots next to your order and select download invoice. The invoice will be downloaded and saved to your device in pdf format. If you need further assistance, write to us on someone@example.com."], 
            'check_payment_methods': ["Currently we accept payments through Credit card, Debit card, Net banking and UPI. For more information regarding payments you can ask specific questions and I will be happy to assist you."],
            'check_refund_policy': ["To check our policy regarding refunds, you can read this blogpost on www.somearticle.com. Also you can write to our payments team on payments@example.com."], 
            'complaint': ["We're truly sorry to hear that you have concerns about our company. We would appreciate it if you could share more details with us, and we'll focus on resolving any issues you're experiencing. Please send a detailed query to someone@example.com, and our team will assist you as soon as possible."], 
            'contact_customer_service': ["To talk to our customer service agent, please write to us on support@example.com. Our team will be contacting you on your registered mobile number of mail as soon as possible."],
            'contact_human_agent': ["To talk to our customer service agent, please write to us on support@example.com. Our team will be contacting you on your registered mobile number of mail as soon as possible."],
            'create_account': ["To create an account, click on the signup button and follow the instructions given on the interface."], 
            'delete_account': ["To delete your account, click on the settings button, then go to Account & Security and click on the 'Delete my Account' button. Then you have to enter your password and click on Schedule to Delete. Once scheduled for deletion your profile will no longer be visible and it will be permanently deleted after 30 days. Please note that you can stop the deletion process within 30 days of scheduling a deletion."],
            'delivery_options': ["Rest assured, we will deliver the product right at your house (or any provided location ofcourse). In case of a refund, our delivery partner will pick the order item from your home so you don't have to worry about how you could return the product. For more information, write to us on someone@example.com"], 
            'delivery_period': ["To check your shipment, go to your profile, select 'Your Orders', then choose the product you want to check shipment status. Click on the three dots on top right and select track my shipment. You will see all the details of your shipment including expected delivery date."], 
            'edit_account': ["To edit your account information, click on the settings button, then go to Account & Security and click on the 'Edit Account' button. There you can make changes regarding your personal information and once done you can click on the 'Save' button to save it."],
            'get_invoice': ["You can download any invoice from your purchase in the 'Your Orders' section. Go to your profile, select 'Your Orders', tap the three dots next to your order and select download invoice. The invoice will be downloaded and saved to your device in pdf format. If you need further assistance, write to us on someone@example.com"], 
            'get_refund': ["To get a refund on any of your purchases you can raise a ticket in the 'Refund' section in the settings. Then you have to mail us at payment@example.com with the reason behind the refund, and our team will assist you. You can check if you are eligible for refund by reading our 'Refund Policies' blog."], 
            'newsletter_subscription': ["To subscribe to our newsletter, click on the link here: randomnewsletterlink.com"],
            'payment_issue': ["For any payment related issue, write to us on payment@example.com and our payments team will reach out to you and help you with your issue."], 
            'place_order': ["You can place order by adding your desired item in the card. Then you have to click on the 'Cart' icon on top right of the app homescreen and click place order. Select the address for delivery, and the desired payment method, complete the payment and your order will be placed. If you still face any issue write to us on someone@example.com. "], 
            'recover_password': ["To reset your password, click on forgot passowrd button in the log in screen. Then enter your registered email or mobile number to receive an one time verification password(OTP). Enter the OTP and you will be able to enter a new password for your account."],
            'registration_problems': ["If you are facing an error in the sign up process, or not receiving an OTP try restarting your device. If the problem persists, contact us on someone@example.com and our team will assist you with the issue."], 
            'review': ["To give us some feedbacks you can rate us on the app store and write your feedback. Our team values every feedback and we constantly are working on improving the user experience with the help of your valuable feedbacks. Additionally you can also fill this google form to give us feedbacks: https://somegoogleform.com"], 
            'set_up_shipping_address': ["To add a shipping address, go to your profile, select 'Saved Address', then click on 'add new address' and fill the details of your address and click 'Save'."],
            'switch_account': ["To switch account, you can go to settings and click 'Log Out'. After logging out, you can then login with a different account by entering email and password."], 
            'track_order': ["To track your shipment, go to your profile, select 'Your Orders', then choose the product you want to check shipment status. Click on the three dots on top right and select track my shipment. You will see all the details of your shipment including expected delivery date"], 
            'track_refund': ["To track status of your refund, go to the 'Refund' section in the settings. Then click on 'Track your ticket' and enter your ticket number and submit. You will get all the updates regarding the refund there."]}

import streamlit as st

# Function to handle chatbot response
def chatbot_response(user_input):
    text = [user_input]

    # text preprossesing
    lemmatizer = WordNetLemmatizer()
    corpus = []

    review = text[0].lower()
    review = review.split()

    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words("english")]
    review = ' '.join(review)
    corpus.append(review)

    # one hot encoding and padding
    one_hot_repr = tokenizer.texts_to_sequences(text)

    # pre padding
    sent_length = 15
    padded_doc = pad_sequences(one_hot_repr, padding='pre', maxlen=sent_length)

    # model prediction
    predictions = model.predict(padded_doc)

    predicted_class = np.argmax(predictions, axis=1)
    a = predicted_class[0]
    intent = labels[a]
    bot = responses[intent]

    return bot[0]

# Create a Streamlit interface
st.title("Chat with your Bot")
st.write("Type your message and interact with the bot")

#NEW 
if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.chat_message("assistant"):
        st.markdown("Hey! How may I help you?")
    st.session_state.messages.append({"role":"assistant","content":"Hey! How may I help you?"})

    

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Accept user input
if prompt := st.chat_input("Ask your Question"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    bot_response = chatbot_response(prompt)


    with st.chat_message("assistant"):
        st.markdown(bot_response)
    
    st.session_state.messages.append({"role": "assistant", "content": bot_response})


       