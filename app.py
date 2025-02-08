import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
 
clf = RandomForestClassifier()
clf.fit(X_train,y_train)

def classification(features):
    pred = clf.predict([features])
    return iris.target_names[pred][0]


def main():
    st.sidebar.title("Navigate Through the pages")
    page = st.sidebar.radio('Select a Page',["Introduction","Use the model"])
    if page == "Introduction":
        st.title("Welcome to my Website")
        st.write("hello this is Bhanuprasad")
    elif page == "Use the model":
        st.title("Iris Classification Model")
        sepal_length = st.number_input("enter the sepal length")
        sepal_width = st.number_input("enter the sepal width")
        petal_length = st.number_input("enter the petal length")
        petal_width = st.number_input("enter the petal width")

        if st.button("Predict"):
            features = [sepal_length,sepal_width,petal_length,petal_width]
            iris_class = classification(features)
            st.write(f"The iris Class is {iris_class}")



if __name__  == "__main__" :
    main()
