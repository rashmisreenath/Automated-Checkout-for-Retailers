
# Automated-Checkout-for-Retailers

An attempt to automate the billing process in retail stores with an inbuilt face recognizer for the customers using Efficient nets and OpenCV

Aim of the project :- 

a)  Face recognition of the customer

b)  Object detection

c)  Bill generation

## Dataset

Dataset :- 
Custom dataset is to be generated, Method used - Web Scraping
Sample dataset Outlook:-
<img width="558" alt="uo1" src="https://user-images.githubusercontent.com/65388338/196435731-569ebb93-e52c-486d-8eca-f8a403cde971.PNG">


## Methodology

a) Efficient Nets was used for the puposes of obejct detection as the EfficientNet models achieve both higher accuracy and better efficiency over existing CNNs, reducing parameter size and FLOPS by an order of magnitude.

b) Live Photo Capture was achieved using OpenCV

c) Bill Generation was achieved as a combination of the following 2 features

d) Streamlit for the purposes of UI
## How to run

cd C:\Desktop\streamlit

streamlitenv\Scripts\activate

streamlit run demo.py
