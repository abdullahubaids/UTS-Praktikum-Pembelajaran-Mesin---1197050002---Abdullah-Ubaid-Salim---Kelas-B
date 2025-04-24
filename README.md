# UTS-Praktikum-Pembelajaran-Mesin---1197050002---Abdullah-Ubaid-Salim---Kelas-B

Dataset : https://drive.google.com/drive/folders/1UDw8t2pVrdaa-b_Sf1JihNr0jnUmM30m?usp=sharing
1.	Persiapan Data
    Load dataset:
    import pandas as pd
    data = pd.read_csv('citrus.csv')

    Pembersihan data:
    -	Periksa dan tangani missing values.
    -	Tidak ada data yang hilang, lanjutkan.

    Konversi data kategorikal:
    -	Ubah kolom name menjadi numerik menggunakan Label Encoding:
      from sklearn.preprocessing import LabelEncoder
      le = LabelEncoder()
      data['name'] = le.fit_transform(data['name'])
    -	orange = 1, grapefruit = 0

    Normalisasi data:
    -	Lakukan penskalaan fitur menggunakan MinMaxScaler:
      from sklearn.preprocessing import MinMaxScaler
      scaler = MinMaxScaler()
      data[['diameter', 'weight', 'red', 'green', 'blue']] = scaler.fit_transform(data[['diameter', 'weight', 'red', 'green', 'blue']])

    Split data:
      from sklearn.model_selection import train_test_split
      X = data.drop('name', axis=1)
      y = data['name']
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
2.	Pelatihan Model
    -	Inisialisasi dan latih model Gaussian Naive Bayes:
      from sklearn.naive_bayes import GaussianNB
      model = GaussianNB()
      model.fit(X_train, y_train)

3.	Evaluasi Model
    -	Prediksi pada data test:
      y_pred = model.predict(X_test)
    -	Evaluasi menggunakan metrik yang sesuai:
      from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
      import seaborn as sns
      import matplotlib.pyplot as plt
      print("Accuracy:", accuracy_score(y_test, y_pred))
      print("\nClassification Report:\n", classification_report(y_test, y_pred))
      cm = confusion_matrix(y_test, y_pred)
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
      plt.xlabel('Predicted')
      plt.ylabel('Actual')
      plt.title('Confusion Matrix')
      plt.show()

4.	Interpretasi Hasil
    -	Akurasi menunjukkan seberapa baik model melakukan klasifikasi secara keseluruhan.
    -	Precision, Recall, dan F1-score memberikan wawasan tentang kinerja model untuk setiap kelas.
    -	Confusion matrix memberikan analisis detail tentang true positives, true negatives, false positives, dan false negatives.

5.	Visualisasi
    -	Visualisasikan matriks confusion untuk mengevaluasi kinerja model.
