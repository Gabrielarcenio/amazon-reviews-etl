import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Baixar stopwords do nltk
nltk.download('stopwords')

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    data = data.drop_duplicates()
    data = data.dropna()
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

if __name__ == "__main__":
    # Caminho para o arquivo CSV
    file_path = "/Users/gabrielarcenio/amazon-reviews-etl/data/raw/Reviews.csv"
    
    # Carregando os dados
    data = load_data(file_path)
    
    # Limpando os dados
    data = clean_data(data)
    
    # Normalizando os dados numéricos
    data = normalize_data(data)
    
    # Processando a coluna de texto (supondo que seja a coluna 'Text')
    data['Text'] = data['Text'].apply(preprocess_text)
    
    # Exibindo as primeiras linhas dos dados transformados
    print(data.head())

    # Criar o diretório 'processed' se não existir
    processed_dir = "/Users/gabrielarcenio/amazon-reviews-etl/data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Salvando os dados transformados em um novo arquivo CSV
    output_file_path = os.path.join(processed_dir, 'transformed_reviews.csv')
    data.to_csv(output_file_path, index=False)
    print(f"Dados transformados salvos em {output_file_path}")
