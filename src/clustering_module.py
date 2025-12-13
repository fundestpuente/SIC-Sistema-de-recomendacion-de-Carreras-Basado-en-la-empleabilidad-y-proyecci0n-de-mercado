import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

class CareerClusterer:
    def __init__(self, df_master):
        self.df = df_master.copy()
        self.features = ['num_estudiantes', 'num_ofertas', 'salario_oferta', 'tasa_empleo_formal']
        self.kmeans = None

    def ejecutar_clustering(self):
        """
        Ejecuta K-Means y aplica la lógica de etiquetado del Notebook 02.
        """
        # 1. Escalar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.features])
        
        # 2. Clustering
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # 3. Etiquetado Inteligente (Lógica exacta del Notebook)
        centroides = self.df.groupby('cluster')[self.features].mean()
        medias = self.df[self.features].mean()

        def _get_label(row, m):
            alta_demanda = row['num_ofertas'] > m['num_ofertas']
            buen_sueldo = row['salario_oferta'] > m['salario_oferta']
            empleo_estable = row['tasa_empleo_formal'] > m['tasa_empleo_formal']
            muchos_graduados = row['num_estudiantes'] > m['num_estudiantes']

            if alta_demanda and buen_sueldo:
                return "En Demanda"
            elif muchos_graduados and (not alta_demanda or not empleo_estable):
                return "Saturadas"
            elif not muchos_graduados and (buen_sueldo or empleo_estable):
                return "Nicho"
            else:
                return "Balanceadas"

        # Aplicar etiquetas a los clusters
        mapa_labels = {i: _get_label(centroides.loc[i], medias) for i in centroides.index}
        self.df['categoria'] = self.df['cluster'].map(mapa_labels)
        
        return self.df

    def plot_3d(self):
        """Genera el Scatter 3D idéntico al del Notebook"""
        fig = px.scatter_3d(
            self.df,
            x='num_estudiantes',
            y='num_ofertas',
            z='salario_oferta',
            color='categoria',
            symbol='categoria',
            hover_name='carrera',
            hover_data={'tasa_empleo_formal':':.1f%', 'sector_economico':True},
            title='<b>Mapa de Clusters 3D</b>',
            labels={
                'num_estudiantes': 'Graduados',
                'num_ofertas': 'Ofertas',
                'salario_oferta': 'Salario ($)'
            },
            color_discrete_map={
                "En Demanda": "#00CC96",
                "Saturadas": "#EF553B",
                "Nicho": "#636EFA",
                "Balanceadas": "#FECB52"
            },
            opacity=0.9,
            height=600
        )
        return fig