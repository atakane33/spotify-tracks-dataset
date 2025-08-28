import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dosya = pd.read_csv("C:/Users/hp/Desktop/nlp/spotify-tracks-dataset/dataset.csv")

#İlk girdi
print("Boyut: ",dosya.shape)
print(dosya.head())
print(dosya.info())
print(dosya.describe())

#Eksik girdi kontrolü
print(dosya.isnull().sum())

#Top 10
print(dosya.isnull().sum().sort_values(ascending=False).head(10))

#Dosya filtreleme
temiz = dosya[(dosya["duration_ms"] > 0) & (dosya["duration_ms"] < 1_200_000)]
print("Arınan bellek boyutu : ", temiz.shape)

#EDA
print(dosya["track_genre"].value_counts().head(10))
print("En uzun şarkı:", dosya.loc[dosya["duration_ms"].idxmax()])
print("En kısa şarkı:", dosya.loc[dosya["duration_ms"].idxmin()])

#Stats
dosya["duration_min"] = dosya["duration_ms"] / 60000
genre_stats = dosya.groupby("track_genre")[["danceability", "energy", "valence", "tempo", "duration_min"]].mean().reset_index()
print(genre_stats.head())

#corr
numeric_cols = dosya.select_dtypes(include=["int64", "float64"])
print(numeric_cols.corr())

#Top 10 Sanatçı filtreleme
topsanatcilar = dosya.groupby("artists")["popularity"].mean().sort_values(ascending=False).head(10)
print(topsanatcilar)

#Top 10 En fazla sarkisi olan sanatcilar
enfazlasarki = dosya["artists"].value_counts().head(10)
print(enfazlasarki)

#More EDA
popsarki = dosya.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(10).reset_index()
longsarki = dosya.groupby("artists")["duration_min"].mean().sort_values(ascending=False).head(10).reset_index()
    #Küfürlü şarkı oranı : (T: 91.45/ F : 8.55)
explicitsarki = dosya["explicit"].value_counts(normalize=True) * 100

print(popsarki)
print(longsarki)
print(explicitsarki)

#Görsel kısım
plt.figure(figsize=(12,6))
sns.barplot(x=topsanatcilar.index,
    y=topsanatcilar.values,
    hue=topsanatcilar.index,
    palette="Blues_r",
    legend=False,
    errorbar=None)
plt.xlabel("Sanatçılar")
plt.ylabel("Ortalama Popülerlik")
plt.title("Top 10 En Popüler Sanatçı (Ortalama Popülerlik)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(x=enfazlasarki.index,
    y=enfazlasarki.values,
    hue=enfazlasarki.index,
    palette="Blues_r",
    legend=False,
    errorbar=None)
plt.xlabel("Sanatçılar")
plt.ylabel("Şarkı Sayısı")
plt.title("Top 10 En Fazla Şarkısı Olan Sanatçı (Ortalama Popülerlik)")
plt.xticks(rotation=45, fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.pie(
    explicitsarki,
    labels=explicitsarki.index.map({True:"Küfürlü", False:"Temiz"}),
    autopct="%.2f%%",
    colors=sns.color_palette("Set2")
)
plt.title("Küfürlü Şarkı Oranı")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Korelasyon Isı Haritası")
plt.show()



