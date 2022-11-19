
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

## How to run these codes

cd C:\Desktop\streamlit
streamlitenv\Scripts\activate
streamlit run demo.py


## NOTE 
Initally YOLO was implemented but as the dataset used was custom dataset, based on the research Efficient nets was found to be the best approch.
This Project was done as a base model/webapp for creating an automated checkout, this could be further improvised based on the requirements of the retail stores, additional items can be added based on the requirement.
(Accuracy increases with the increase in training set)


## References

https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html#:~:text=EfficientNet%20Performance&text=In%20general%2C%20the%20EfficientNet%20models,by%20an%20order%20of%20magnitude.
