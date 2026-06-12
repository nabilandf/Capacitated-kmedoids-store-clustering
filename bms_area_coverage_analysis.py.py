# -*- coding: utf-8 -*-
"""
Capacitated K-Medoids Store Clustering
Author: Nabila Nadhifatuzzahra

Store territory optimization using Capacitated K-Medoids clustering.
"""

# =====================
# IMPORT LIBRARIES
# =====================

import pandas as pd
import numpy as np
import folium
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial import ConvexHull
from folium import FeatureGroup, LayerControl

# =====================
# LOAD DATA
# =====================

# Dataset is not included in this repository
# due to confidentiality reasons.
# Expected columns:
# - STORE_ID
# - STORE_NAME
# - LATITUDE
# - LONGITUDE
# - BRANCH

# Example:
# df = pd.read_csv("store_data.csv")


# =====================
# FILTER DATA (CABANG DISESUAIKAN)
# =====================
branches = [
    "BRANCH_A",
    "BRANCH_B",
    "BRANCH_C",
    "BRANCH_D"
]

df['BRANCH'] = df['BRANCH'].astype(str).str.upper()
df_jabo = df[df["BRANCH"].isin(branches)].copy().reset_index(drop=True)

# ambil kolom penting (buang semua pool lama)
df_jabo = df_jabo[
[
'STORE_ID',
'STORE_NAME',
'LATITUDE',
'LONGITUDE',
'BRANCH'
]
].dropna().reset_index(drop=True)

print("Total toko:", len(df_jabo))

#CEK JUMLAH TOKO TIAP CABANG
print(df_jabo["BRANCH"].value_counts())
print("Total toko:", len(df_jabo))

# =====================
# PARAMETER
# =====================
k = 65 # Adjust according to the desired number of clusters

n = len(df_jabo)
base = n // k 
extra = n - k * base

#Validasi
assert n == (extra * (base + 1) + (k - extra) * base)

# =====================
# DISTANCE CALCULATION (HAVERSINE)
# =====================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

# =====================
# CAPACITY
# =====================
import random

def generate_capacity(k, base, extra):
    capacity = [base]*k
    idx = list(range(k))
    random.shuffle(idx)
    for i in idx[:extra]:
        capacity[i] += 1
    return capacity

# =====================
# ASSIGN CLUSTER
# =====================
def assign_clusters(df, medoids, capacity):
    clusters = {i: [] for i in range(len(medoids))}

    for idx, row in df.iterrows():
        distances = []

        for j, m in medoids.iterrows():
            d = haversine(row['LATITUDE'], row['LONGITUDE'],
                          m['LATITUDE'], m['LONGITUDE'])
            distances.append((j, d))
        distances.sort(key=lambda x: x[1])

        for j, _ in distances:
            if len(clusters[j]) < capacity[j]:
                clusters[j].append(idx)
                break

    return clusters

# =====================
# UPDATE MEDOID
# =====================

def update_medoids(df, clusters):
    new_medoids = []

    for c, members in clusters.items():
        sub = df.loc[members].copy()

        lat_mean = sub['LATITUDE'].mean()
        lon_mean = sub['LONGITUDE'].mean()

        sub['dist'] = sub.apply(lambda x: haversine(
            x['LATITUDE'], x['LONGITUDE'],
            lat_mean, lon_mean
        ), axis=1)

        best_idx = sub['dist'].idxmin()
        new_medoids.append(df.loc[best_idx])

    return pd.DataFrame(new_medoids).reset_index(drop=True)

# =====================
# RUN CLUSTERING
# =====================
def run_clustering(df, k, base, extra, n_iter=5, seed=42):
    np.random.seed(seed)
    random.seed(seed)

#MEMILIH MEDOID AWAL (TITIK PUSAT SEMENTARA)
    medoid_idx = np.random.choice(df.index, size=k, replace=False)
    medoids = df.loc[medoid_idx].copy().reset_index(drop=True)

    capacity = generate_capacity(k, base, extra)

    for _ in range(n_iter):
        clusters = assign_clusters(df, medoids, capacity)
        medoids = update_medoids(df, clusters)

    # Calculate total distance after clustering
    dist = total_distance(df, clusters, medoids)

    return clusters, medoids, dist

# =====================
# TOTAL DISTANCE
# =====================

def total_distance(df, clusters, medoids):
    total = 0

    # Loop setiap cluster
    for c, members in clusters.items():
        m = medoids.loc[c] 

        for i in members:
            row = df.loc[i]

            total += haversine(row['LATITUDE'], row['LONGITUDE'],
                               m['LATITUDE'], m['LONGITUDE'])

    return total

# =====================
# MULTI RUN
# =====================
best_dist = float('inf')

for seed in range(10):
    print(f"Run {seed}")

    clusters, medoids, dist = run_clustering(df_jabo, k, base, extra, seed=seed)

    print("Distance:", dist)

    if dist < best_dist:
        best_dist = dist
        best_clusters = clusters
        best_medoids = medoids

print("BEST:", best_dist)

#MENAMPILKAN TOKO POOL
best_clusters
best_medoids

# =====================
# ASSIGN CLUSTER
# =====================
df_jabo['CLUSTER'] = -1

for c, members in best_clusters.items():
    for m in members:
        df_jabo.loc[m, 'CLUSTER'] = c

# =====================
# BUAT POOL BARU
# =====================
best_medoids = best_medoids.copy()
best_medoids['CLUSTER'] = best_medoids.index

pool_map = best_medoids.rename(columns={
    'STORE_ID': 'KODE TOKO_POOL',
    'STORE_NAME': 'NAMA TOKO_POOL',
    'LATITUDE': 'LATITUDE_POOL',
    'LONGITUDE': 'LONGITUDE_POOL'
})

# =====================
# MERGE POOL
# =====================
df_final = df_jabo.merge(
    pool_map[['CLUSTER','KODE TOKO_POOL','NAMA TOKO_POOL','LATITUDE_POOL','LONGITUDE_POOL']],
    on='CLUSTER',
    how='left'
)

# =====================
# HITUNG JARAK KE POOL
# =====================
df_final['DIS_to_POOL (km)'] = df_final.apply(
    lambda x: haversine(
        x['LATITUDE'], x['LONGITUDE'],
        x['LATITUDE_POOL'], x['LONGITUDE_POOL']
    ),
    axis=1
)

# =====================
# TANDAI TOKO POOL
# =====================
df_final['POOL'] = ''

pool_codes = set(pool_map['KODE TOKO_POOL'])
df_final.loc[df_final['KODE TOKO'].isin(pool_codes), 'POOL'] = 'TOKO POOL'

# =====================
# VALIDASI JUMLAH TOKO
# =====================
sizes = df_final.groupby("CLUSTER").size()
print("MIN:", sizes.min(), "MAX:", sizes.max())

#OUTPUT DATA YANG INGIN DITAMPILKAN
df_final = df_final[[
    'CLUSTER',
    'BRANCH',
    'KODE TOKO',
    'NAMA TOKO',
    'LATITUDE',
    'LONGITUDE',
    'KODE TOKO_POOL',
    'NAMA TOKO_POOL',
    'LATITUDE_POOL',
    'LONGITUDE_POOL',
    'DIS_to_POOL (km)',
    'POOL'
]]


# Save clustering result
df_final.to_csv("clustering_results.csv", index=False)

# =====================
# VISUALIZATION
# =====================

BRANCH_COLOR = {
    "BRANCH_A": "#FFD700",      # kuning
    "BRANCH_B": "#FF0000",    # merah
    "BRANCH_C": "#FF8C00",     # orange
    "BRANCH_D": "#00CED1",      # cyan
}

# =====================
# WARNA CLUSTER
# =====================
COLOR_LIST = [
    "#FF0000",
    "#0000FF",
    "#00AA00",
    "#FF8C00",
    "#800080",
    "#00CED1",
    "#FF1493",
    "#8B4513",
    "#FFD700",
    "#2F4F4F",
    "#DC143C",
    "#4B0082",
    "#228B22",
    "#FF4500",
    "#008B8B"
]

# =====================
# CONVEX HULL
# =====================
def get_hull(coords):

    if len(coords) < 3:
        return coords

    hull = ConvexHull(coords)

    return coords[hull.vertices]

# =====================
# MAPPING MEDOID
# =====================
best_medoids = best_medoids.copy()
best_medoids["CLUSTER"] = best_medoids.index

medoid_map = {
    row["CLUSTER"]: row
    for _, row in best_medoids.iterrows()
}

# =====================
# MAP FUNCTION
# =====================
def plot_map(df):

    # =====================
    # BASE MAP
    # =====================
    m = folium.Map(
        location=[
            df["LATITUDE"].mean(),
            df["LONGITUDE"].mean()
        ],
        zoom_start=10
    )

    # =====================
    # GLOBAL LAYER
    # =====================
    layer_branch = folium.FeatureGroup(name="Cabang")
    layer_cluster_area = folium.FeatureGroup(name="Cluster Area")

    # =====================
    # POLYGON CABANG
    # =====================
    for branch, group in df.groupby("BRANCH"):

        coords = group[["LATITUDE", "LONGITUDE"]].values

        if len(coords) >= 3:

            try:

                hull = get_hull(coords)

                popup_text = folium.Popup(
                    f"""
                    <b>CABANG:</b> {branch}<br>
                    <b>Jumlah Toko:</b> {len(group)}
                    """,
                    max_width=250
                )

                folium.Polygon(
                    locations=hull,
                    color=BRANCH_COLOR.get(branch, "black"),
                    weight=3,
                    fill=True,
                    fill_color=BRANCH_COLOR.get(branch, "black"),
                    fill_opacity=0.08,
                    popup=popup_text,
                    interactive=True
                ).add_to(layer_branch)

            except:
                pass

    # =====================
    # CLUSTER LOOP
    # =====================
    for cluster, group in df.groupby("CLUSTER"):

        # warna cluster lebih strong
        color = COLOR_LIST[
            cluster % len(COLOR_LIST)
        ]

        coords = group[["LATITUDE", "LONGITUDE"]].values

        # =====================
        # LAYER KHUSUS CLUSTER
        # =====================
        layer_cluster_toko = folium.FeatureGroup(
            name=f"POOL {cluster}",
            show=False
        )

        # =====================
        # POLYGON CLUSTER
        # =====================
        if len(coords) >= 3:

            try:

                hull = get_hull(coords)

                folium.Polygon(
                    locations=hull,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.12,
                    weight=2,
                    interactive=False
                ).add_to(layer_cluster_area)

            except:
                pass

        # =====================
        # MEDOID
        # =====================
        med = medoid_map[cluster]

        # =====================
        # TITIK TOKO
        # =====================
        for _, row in group.iterrows():

            popup_toko = folium.Popup(
                f"""
                <b>Cabang:</b> {row['BRANCH']}<br>
                <b>Kode Toko:</b> {row['KODE TOKO']}<br>
                <b>Nama Toko:</b> {row['NAMA TOKO']}<br>
                <b>Cluster:</b> {cluster}<br>
                <b>Pool:</b> {med['KODE TOKO']} - {med['NAMA TOKO']}<br>
                <b>Jarak:</b> {row['DIS_to_POOL (km)']:.2f} km
                """,
                max_width=250
            )

            folium.CircleMarker(
                location=[
                    row["LATITUDE"],
                    row["LONGITUDE"]
                ],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1,
                weight=1,
                popup=popup_toko
            ).add_to(layer_cluster_toko)

        # =====================
        # MARKER MEDOID
        # =====================
        popup_pool = folium.Popup(
            f"""
            <b>POOL (MEDOID)</b><br>
            <b>Cluster:</b> {cluster}<br>
            <b>Kode:</b> {med['KODE TOKO']}<br>
            <b>Nama:</b> {med['NAMA TOKO']}<br>
            <b>Cabang:</b> {med['BRANCH']}
            """,
            max_width=250
        )

        folium.Marker(
            location=[
                med["LATITUDE"],
                med["LONGITUDE"]
            ],
            popup=popup_pool,
            icon=folium.Icon(
                color="red",
                icon="home"
            )
        ).add_to(layer_cluster_toko)

        # =====================
        # ADD CLUSTER LAYER
        # =====================
        layer_cluster_toko.add_to(m)

    # =====================
    # ADD GLOBAL LAYER
    # =====================
    layer_cluster_area.add_to(m)
    layer_branch.add_to(m)

    # =====================
    # LAYER CONTROL
    # =====================
    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    return m

# =====================
# RUN
# =====================
m = plot_map(df_final)

m.save("cluster_map.html.html")
print("DONE")
m
