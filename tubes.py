import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisis Performa Siswa", layout="wide")

st.title(" Analisis Performa Siswa - Data Matematika")

st.markdown("""
Aplikasi ini menampilkan visualisasi interaktif untuk memahami karakteristik dan performa siswa berdasarkan data dari UCI Student Performance Dataset.
""")

# Load data
@st.cache_data
def load_datasets():
    mat = pd.read_csv("student-mat.csv", sep=";")
    por = pd.read_csv("student-por.csv", sep=";")
    mat["Dataset"] = "Math"
    por["Dataset"] = "Portuguese"
    combined = pd.concat([mat, por])
    return mat, por, combined

df_mat, df_por, df_all = load_datasets()

# Sidebar Select Dataset
st.sidebar.header(" Dataset")
dataset_option = st.sidebar.selectbox("Select Dataset", options=["All", "Math", "Portuguese"])
if dataset_option == "Math":
    df = df_mat
elif dataset_option == "Portuguese":
    df = df_por
else:
    df = df_all

# Sidebar Filters
st.sidebar.header(" Filter Data")
selected_gender = st.sidebar.multiselect("Pilih Gender:", options=df["sex"].unique(), default=df["sex"].unique())
selected_school = st.sidebar.multiselect("Pilih Sekolah:", options=df["school"].unique(), default=df["school"].unique())

filtered_df = df[(df["sex"].isin(selected_gender)) & (df["school"].isin(selected_school))]

# Preview Dataset
with st.expander("Preview Dataset"):
    st.dataframe(filtered_df.head())

with st.expander("Descriptive Statistics"):
    st.write(filtered_df.describe())
    st.write(f"Jumlah data: {filtered_df.shape[0]} baris dan {filtered_df.shape[1]} kolom")

# ============ HYBRID SECTION 1: SUMMARY STATISTICS (OVERVIEW ALL GRADES) ============
st.header(" Summary Statistics - Overview Semua Nilai")

col1, col2 = st.columns(2)

with col1:
    # Bar chart untuk rata-rata semua grades
    avg_grades = pd.DataFrame({
        'Grade': ['G1', 'G2', 'G3'],
        'Average': [filtered_df['G1'].mean(), filtered_df['G2'].mean(), filtered_df['G3'].mean()],
        'Std': [filtered_df['G1'].std(), filtered_df['G2'].std(), filtered_df['G3'].std()]
    })
    
    fig_avg = go.Figure()
    fig_avg.add_trace(go.Bar(
        x=avg_grades['Grade'],
        y=avg_grades['Average'],
        error_y=dict(type='data', array=avg_grades['Std']),
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        text=avg_grades['Average'].round(2),
        textposition='outside'
    ))
    fig_avg.update_layout(
        title="Rata-rata Nilai G1, G2, G3",
        yaxis_title="Rata-rata Nilai",
        showlegend=False
    )
    st.plotly_chart(fig_avg, use_container_width=True)

with col2:
    # Box plot perbandingan distribusi semua grades
    grades_melted = pd.melt(filtered_df[['G1', 'G2', 'G3']], var_name='Grade', value_name='Score')
    fig_box_all = px.box(grades_melted, x='Grade', y='Score', color='Grade',
                         title="Distribusi Semua Nilai (G1, G2, G3)",
                         color_discrete_map={'G1': '#FF6B6B', 'G2': '#4ECDC4', 'G3': '#45B7D1'})
    st.plotly_chart(fig_box_all, use_container_width=True)

# Correlation heatmap untuk G1, G2, G3
st.subheader(" Korelasi antar Nilai")
corr_matrix = filtered_df[['G1', 'G2', 'G3']].corr()
fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Heatmap Korelasi G1, G2, G3",
                        color_continuous_scale='RdYlBu_r')
st.plotly_chart(fig_heatmap, use_container_width=True)

# ============ BASIC DEMOGRAPHICS ============
st.header(" Demografi Siswa")

col1, col2 = st.columns(2)

with col1:
    # Gender distribution - Donut Chart
    st.subheader("Distribusi Gender")
    gender_counts = filtered_df['sex'].value_counts()
    fig_gender = go.Figure(data=[
        go.Pie(labels=gender_counts.index, values=gender_counts.values, hole=0.5,
               marker=dict(colors=["#00cc96", "#ffa600"]),
               textinfo='label+value+percent')
    ])
    fig_gender.update_layout(
        title_text="Persentase Gender Siswa",
        annotations=[dict(text=f"{filtered_df.shape[0]}<br>students", x=0.5, y=0.5, font_size=18, showarrow=False)]
    )
    st.plotly_chart(fig_gender, use_container_width=True)

with col2:
    # Area distribution
    st.subheader("Distribusi Area Tempat Tinggal")
    area_counts = filtered_df['address'].value_counts()
    fig_area = go.Figure(data=[go.Pie(
        labels=area_counts.index,
        values=area_counts.values,
        hole=0.4,
        marker=dict(colors=['#E9762B', '#0D4715']),
        textinfo='label+percent'
    )])
    fig_area.update_layout(
        title='Area Tempat Tinggal',
        showlegend=False,
        annotations=[dict(text='Area', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    st.plotly_chart(fig_area, use_container_width=True)

# Age histogram
st.subheader("Distribusi Usia Siswa")
fig_age = px.histogram(filtered_df, x="age", nbins=10, title="Histogram Usia Siswa")
st.plotly_chart(fig_age, use_container_width=True)

# ============ HYBRID SECTION 2: ACADEMIC PERFORMANCE WITH TOGGLE ============
st.header(" Analisis Performa Akademik")

# Toggle untuk menampilkan semua grades atau pilih spesifik
show_all_grades = st.checkbox(" Tampilkan Semua Nilai (G1, G2, G3)", value=True)

if show_all_grades:
    # Tampilkan semua grades
    st.subheader(" Distribusi Semua Nilai")
        
    
    # Histogram untuk semua grades
    fig_hist_all = go.Figure()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, grade in enumerate(['G1', 'G2', 'G3']):
        fig_hist_all.add_trace(go.Histogram(
            x=filtered_df[grade],
             name=grade,
            opacity=0.7,
            marker_color=colors[i]
        ))
    fig_hist_all.update_layout(
        title="Distribusi Semua Nilai",
        barmode='overlay',
        xaxis_title="Nilai",
        yaxis_title="Frekuensi"
    )
    st.plotly_chart(fig_hist_all, use_container_width=True)
    
else:
    # Pilih grade spesifik untuk analisis detail
    selected_grade = st.selectbox("Pilih Nilai untuk Analisis:", options=["G1", "G2", "G3"], index=2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram untuk grade yang dipilih
        fig_grade = px.histogram(filtered_df, x=selected_grade, color="sex", barmode="overlay", 
                                title=f"Distribusi {selected_grade}")
        st.plotly_chart(fig_grade, use_container_width=True)
    
    with col2:
        # Boxplot berdasarkan sekolah
        fig_box = px.box(filtered_df, x="school", y=selected_grade, color="school", 
                        title=f"Boxplot {selected_grade} per Sekolah")
        st.plotly_chart(fig_box, use_container_width=True)

# ============ HYBRID SECTION 3: STUDY PATTERNS WITH INTERACTIVE CONTROLS ============
st.header(" Pola Belajar dan Dampaknya")

# Toggle untuk analisis study time
study_analysis_type = st.radio(
    "Pilih Jenis Analisis:",
    ["Overview Semua Nilai", "Analisis Detail per Nilai", "Perbandingan Faktor"]
)

if study_analysis_type == "Overview Semua Nilai":
    # Tampilkan dampak study time terhadap semua grades
    st.subheader(" Dampak Waktu Belajar terhadap Semua Nilai")
    
    studytime_labels = {1: '<2 jam', 2: '2-5 jam', 3: '5-10 jam', 4: '>10 jam'}
    filtered_df['studytime_label'] = filtered_df['studytime'].map(studytime_labels)
    
    # Grouped bar chart untuk semua grades
    study_avg = filtered_df.groupby('studytime_label')[['G1', 'G2', 'G3']].mean().reset_index()
    study_melted = pd.melt(study_avg, id_vars=['studytime_label'], 
                          value_vars=['G1', 'G2', 'G3'], 
                          var_name='Grade', value_name='Average')
    
    fig_study_all = px.bar(study_melted, x='studytime_label', y='Average', 
                          color='Grade', barmode='group',
                          title="Rata-rata Semua Nilai berdasarkan Waktu Belajar",
                          color_discrete_map={'G1': '#FF6B6B', 'G2': '#4ECDC4', 'G3': '#45B7D1'})
    st.plotly_chart(fig_study_all, use_container_width=True)

elif study_analysis_type == "Analisis Detail per Nilai":
    # Analisis detail untuk grade tertentu
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_grade_study = st.selectbox("Pilih Nilai:", options=["G1", "G2", "G3"], 
                                          index=2, key="study_grade")
        chart_type = st.selectbox("Jenis Chart:", options=["Scatter Plot", "Box Plot", "Violin Plot"])
    
    with col2:
        if chart_type == "Scatter Plot":
            fig_study_detail = px.scatter(filtered_df, x="studytime", y=selected_grade_study, 
                                        color="sex", size="absences",
                                        hover_data=["age", "school"], 
                                        title=f"Waktu Belajar vs {selected_grade_study}")
        elif chart_type == "Box Plot":
            fig_study_detail = px.box(filtered_df, x="studytime", y=selected_grade_study, 
                                    color="sex", title=f"Distribusi {selected_grade_study} per Waktu Belajar")
        else:  # Violin Plot
            fig_study_detail = px.violin(filtered_df, x="studytime", y=selected_grade_study, 
                                       color="sex", box=True,
                                       title=f"Distribusi {selected_grade_study} per Waktu Belajar")
        
        st.plotly_chart(fig_study_detail, use_container_width=True)

else:  # Perbandingan Faktor
    # Analisis gabungan studytime dan failures
    st.subheader(" Waktu Belajar vs Kegagalan vs Nilai")
    
    # Pilih grade untuk analisis
    grade_for_comparison = st.selectbox("Pilih Nilai untuk Perbandingan:", 
                                       options=["G1", "G2", "G3"], index=2, key="comparison_grade")
    
    grouped = filtered_df.groupby(['studytime', 'failures'])[grade_for_comparison].mean().reset_index()
    studytime_map = {1: '<2 jam', 2: '2-5 jam', 3: '5-10 jam', 4: '>10 jam'}
    grouped['studytime_str'] = grouped['studytime'].map(studytime_map)
    
    fig_comparison = px.bar(grouped, x='studytime_str', y=grade_for_comparison, 
                           color='failures', barmode='group',
                           title=f'Rata-rata {grade_for_comparison} berdasarkan Waktu Belajar dan Kegagalan')
    st.plotly_chart(fig_comparison, use_container_width=True)

# ============ SUPPORT SYSTEMS ANALYSIS ============
st.header(" Sistem Dukungan")

# Support analysis dengan toggle
support_toggle = st.checkbox(" Tampilkan Analisis untuk Semua Nilai", value=False, key="support_toggle")

support_features = ['schoolsup', 'famsup', 'paid', 'higher']

if support_toggle:
    # Analisis untuk semua grades
    st.subheader(" Pengaruh Dukungan terhadap Semua Nilai")
    
    support_data_all = []
    for feature in support_features:
        for grade in ['G1', 'G2', 'G3']:
            grouped = filtered_df.groupby(feature)[grade].mean().reset_index()
            for _, row in grouped.iterrows():
                support_data_all.append({
                    'Support': feature,
                    'Grade': grade,
                    'Type': row[feature],
                    'Avg_Score': row[grade]
                })
    
    support_df_all = pd.DataFrame(support_data_all)
    
    fig_support_all = px.bar(
        support_df_all,
        x='Support',
        y='Avg_Score',
        color='Type',
        facet_col='Grade',
        barmode='group',
        title='Pengaruh Dukungan terhadap Semua Nilai',
        color_discrete_map={'yes': '#2ecc71', 'no': '#bdc3c7'}
    )
    st.plotly_chart(fig_support_all, use_container_width=True)

else:
    # Analisis untuk G3 saja (default)
    st.subheader(" Pengaruh Dukungan terhadap Nilai Akhir (G3)")
    
    support_data = []
    for feature in support_features:
        grouped = filtered_df.groupby(feature)['G3'].mean().reset_index()
        for _, row in grouped.iterrows():
            support_data.append({
                'Support': feature,
                'Type': row[feature],
                'Avg_G3': row['G3']
            })

    support_data = pd.DataFrame(support_data)

    fig_support = px.bar(
        support_data,
        x='Support',
        y='Avg_G3',
        color='Type',
        barmode='group',
        title='Pengaruh Dukungan terhadap Nilai Akhir (G3)',
        color_discrete_map={'yes': '#2ecc71', 'no': '#bdc3c7'},
        text='Avg_G3'
    )
    fig_support.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig_support, use_container_width=True)

# ============ LIFESTYLE ANALYSIS ============
st.header(" Gaya Hidup dan Dampaknya")

# Lifestyle analysis dengan pilihan grade
col1, col2 = st.columns([1, 3])

with col1:
    lifestyle_grade = st.selectbox("Pilih Nilai untuk Analisis Gaya Hidup:", 
                                  options=["G1", "G2", "G3"], index=2, key="lifestyle_grade")
    show_individual = st.checkbox("Tampilkan Grafik Terpisah", value=False)

with col2:
    if show_individual:
        # Tampilkan dalam subplot terpisah
        lifestyle_factors = ['goout', 'freetime', 'Walc', 'Dalc']
        fig_lifestyle_sep = go.Figure()
        
        for i, factor in enumerate(lifestyle_factors):
            factor_avg = filtered_df.groupby(factor)[lifestyle_grade].mean()
            fig_lifestyle_sep.add_trace(go.Scatter(
                x=factor_avg.index,
                y=factor_avg.values,
                mode='lines+markers',
                name=factor,
                line=dict(width=3)
            ))
        
        fig_lifestyle_sep.update_layout(
            title=f"Gaya Hidup dan Pengaruhnya terhadap {lifestyle_grade}",
            xaxis_title="Level (1=rendah, 5=tinggi)",
            yaxis_title=f"Rata-rata {lifestyle_grade}"
        )
        st.plotly_chart(fig_lifestyle_sep, use_container_width=True)
    else:
        # Tampilkan gabungan seperti aslinya
        goout_avg = filtered_df.groupby('goout', as_index=False)[lifestyle_grade].mean()
        goout_avg['Faktor'] = 'Goout'
        freetime_avg = filtered_df.groupby('freetime', as_index=False)[lifestyle_grade].mean()
        freetime_avg['Faktor'] = 'Freetime'
        walc_avg = filtered_df.groupby('Walc', as_index=False)[lifestyle_grade].mean()
        walc_avg['Faktor'] = 'Walc'
        dalc_avg = filtered_df.groupby('Dalc', as_index=False)[lifestyle_grade].mean()
        dalc_avg['Faktor'] = 'Dalc'
        
        lifestyle_df = pd.concat([
            goout_avg.rename(columns={'goout': 'Level', lifestyle_grade: 'Score'}),
            freetime_avg.rename(columns={'freetime': 'Level', lifestyle_grade: 'Score'}),
            walc_avg.rename(columns={'Walc': 'Level', lifestyle_grade: 'Score'}),
            dalc_avg.rename(columns={'Dalc': 'Level', lifestyle_grade: 'Score'})
        ])
        
        fig_lifestyle = px.line(
            lifestyle_df,
            x="Level", y="Score", color="Faktor", markers=True,
            title=f"Gaya Hidup dan Pengaruhnya terhadap {lifestyle_grade}",
            labels={"Level": "Tingkat", "Score": f"Rata-rata {lifestyle_grade}"}
        )
        st.plotly_chart(fig_lifestyle, use_container_width=True)

# ============ FAMILY & SOCIAL RELATIONSHIPS ============
st.header(" Hubungan Keluarga & Sosial")

col1, col2 = st.columns(2)

with col1:
    # Romantic status analysis
    relationship_grade = st.selectbox("Pilih Nilai untuk Analisis Hubungan:", 
                                     options=["G1", "G2", "G3"], index=2, key="relationship_grade")
    
    fig_romantic = px.violin(
        filtered_df, x='romantic', y=relationship_grade,
        box=True, points='suspectedoutliers',
        color='romantic',
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f'Status Romantis vs {relationship_grade}'
    )
    st.plotly_chart(fig_romantic, use_container_width=True)

with col2:
    # Family relationship quality
    famrel_grade = st.selectbox("Pilih Nilai untuk Analisis Keluarga:", 
                               options=["G1", "G2", "G3"], index=2, key="famrel_grade")
    
    famrel_avg = filtered_df.groupby('famrel')[famrel_grade].mean().reset_index()
    fig_famrel = px.bar(
        famrel_avg, x='famrel', y=famrel_grade, text=famrel_grade,
        title=f'Kualitas Hubungan Keluarga vs {famrel_grade}',
        color=famrel_grade, color_continuous_scale='Greens'
    )
    fig_famrel.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig_famrel, use_container_width=True)

# ============ PARENTAL BACKGROUND ============
st.header(" Latar Belakang Orang Tua")

# Parental education with grade selection
col1, col2 = st.columns([1, 3])

with col1:
    parent_grade = st.selectbox("Pilih Nilai untuk Analisis Orang Tua:", 
                               options=["G1", "G2", "G3"], index=2, key="parent_grade")
    parent_analysis = st.radio("Jenis Analisis:", ["Pendidikan", "Pekerjaan"], key="parent_analysis")

with col2:
    if parent_analysis == "Pendidikan":
        # Education analysis
        fig_edu = px.box(
            pd.concat([
                filtered_df[['Medu', parent_grade]].rename(columns={'Medu': 'Pendidikan', parent_grade: 'Nilai'}).assign(Ortu='Ibu'),
                filtered_df[['Fedu', parent_grade]].rename(columns={'Fedu': 'Pendidikan', parent_grade: 'Nilai'}).assign(Ortu='Ayah')
            ]),
            x='Pendidikan', y='Nilai', color='Ortu',
            title=f'Tingkat Pendidikan Orang Tua vs {parent_grade}'
        )
        st.plotly_chart(fig_edu, use_container_width=True)
    else:
        # Job analysis
        job_df = pd.concat([
            filtered_df[['Mjob', parent_grade]].rename(columns={'Mjob': 'Pekerjaan', parent_grade: 'Nilai'}).assign(Ortu='Ibu'),
            filtered_df[['Fjob', parent_grade]].rename(columns={'Fjob': 'Pekerjaan', parent_grade: 'Nilai'}).assign(Ortu='Ayah')
        ])
        
        fig_job = px.bar(
            job_df.groupby(['Ortu', 'Pekerjaan'])['Nilai'].mean().reset_index(),
            x='Pekerjaan', y='Nilai', color='Ortu', barmode='group', text='Nilai',
            title=f'Pekerjaan Orang Tua vs {parent_grade}'
        )
        fig_job.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig_job, use_container_width=True)

# ============ ADVANCED CORRELATIONS ============
st.header(" Analisis Korelasi Lanjutan")

# Advanced correlation analysis
correlation_type = st.selectbox(
    "Pilih Jenis Analisis Korelasi:",
    ["Hubungan G1-G2-G3", "Faktor vs Semua Nilai", "Analisis Prediktif"]
)

if correlation_type == "Hubungan G1-G2-G3":
    
    # Progression analysis
    progression_data = []
    for idx, row in filtered_df.iterrows():
        progression_data.extend([
            {'Student': idx, 'Period': 'G1', 'Score': row['G1']},
            {'Student': idx, 'Period': 'G2', 'Score': row['G2']},
            {'Student': idx, 'Period': 'G3', 'Score': row['G3']}
        ])
        
    progression_df = pd.DataFrame(progression_data)
    sample_students = progression_df['Student'].unique()[:10]  # Sample 10 students
    sample_df = progression_df[progression_df['Student'].isin(sample_students)]
        
    fig_progression = px.line(sample_df, x='Period', y='Score', 
                                 color='Student', title="Progression Sample Siswa")
    st.plotly_chart(fig_progression, use_container_width=True)

elif correlation_type == "Faktor vs Semua Nilai":
    # Multi-factor correlation
    factor_choice = st.selectbox("Pilih Faktor:", 
                                ["studytime", "failures", "absences", "freetime", "goout"])
    
    fig_multi = go.Figure()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, grade in enumerate(['G1', 'G2', 'G3']):
        correlation_data = filtered_df.groupby(factor_choice)[grade].mean()
        fig_multi.add_trace(go.Scatter(
            x=correlation_data.index,
            y=correlation_data.values,
            mode='lines+markers',
            name=grade,
            line=dict(color=colors[i], width=3),
            marker=dict(size=8)
        ))
    
    fig_multi.update_layout(
        title=f"Pengaruh {factor_choice} terhadap Semua Nilai",
        xaxis_title=factor_choice,
        yaxis_title="Rata-rata Nilai"
    )
    st.plotly_chart(fig_multi, use_container_width=True)

else:  # Analisis Prediktif
    # Predictive analysis visualization
    st.subheader(" Analisis Prediktif G3 berdasarkan G1 & G2")
    
    # Scatter dengan trend line
    fig_pred = px.scatter(filtered_df, x='G1', y='G3', color='G2', 
                         trendline='ols', hover_data=['G2', 'studytime'],
                         title="Prediksi G3 berdasarkan G1 (warna = G2)")
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Residual analysis
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    X = filtered_df[['G1', 'G2']].values
    y = filtered_df['G3'].values
    
    # Remove any NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) > 0:
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        predictions = model.predict(X_clean)
        residuals = y_clean - predictions
        
        fig_residual = px.scatter(x=predictions, y=residuals,
                                 title="Analisis Residual (Predicted vs Residual)",
                                 labels={'x': 'Predicted G3', 'y': 'Residuals'})
        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_residual, use_container_width=True)

# ============ INSIGHTS SECTION ============
st.header(" Insights & Kesimpulan")

with st.expander(" Key Insights", expanded=True):
    
    # Calculate some key statistics
    avg_g3 = filtered_df['G3'].mean()
    high_performers = filtered_df[filtered_df['G3'] >= 15].shape[0]
    total_students = filtered_df.shape[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rata-rata G3", f"{avg_g3:.2f}", "")
        
    with col2:
        pass_rate = (filtered_df['G3'] >= 10).sum() / len(filtered_df) * 100
        st.metric(
        label="Pass Rate (G3≥10)",
        value=f"{pass_rate:.1f}%",
       delta=None
    )
        
    with col3:
        correlation_g1_g3 = filtered_df['G1'].corr(filtered_df['G3'])
        st.metric("Korelasi G1-G3", f"{correlation_g1_g3:.3f}", "")
    
    st.markdown("""
    ###  Insights Utama:
    
    **Performa Akademik:**
    - Terdapat korelasi kuat antara G1, G2, dan G3 - nilai awal dapat memprediksi performa akhir
    - Siswa dengan waktu belajar lebih tinggi cenderung memiliki nilai yang lebih baik
    - Jumlah kegagalan sebelumnya berkorelasi negatif dengan nilai akhir
    
    **Faktor Dukungan:**
    - Dukungan keluarga dan sekolah menunjukkan dampak positif terhadap performa
    - Siswa yang berencana melanjutkan pendidikan tinggi memiliki rata-rata nilai lebih tinggi
    - Dukungan finansial (paid classes) tidak selalu berkorelasi dengan nilai lebih tinggi
    
    **Aspek Sosial & Gaya Hidup:**
    - Konsumsi alkohol berlebihan berkorelasi dengan penurunan nilai
    - Waktu luang yang berlebihan tanpa struktur dapat menurunkan performa
    - Status hubungan romantis memiliki dampak bervariasi pada nilai
    
    **Latar Belakang Keluarga:**
    - Tingkat pendidikan orang tua berkorelasi positif dengan performa siswa
    - Pekerjaan orang tua di bidang pendidikan/kesehatan cenderung menghasilkan nilai lebih tinggi
    - Kualitas hubungan keluarga (famrel) sangat berpengaruh terhadap prestasi akademik
    """)

# ============ INTERACTIVE EXPLORATION SECTION ============
st.header(" Eksplorasi Interaktif")

st.markdown("Bagian ini memungkinkan Anda untuk melakukan eksplorasi data secara bebas:")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Pengaturan Eksplorasi")
    
    # X-axis selection
    x_axis = st.selectbox(
        "Pilih variabel X:",
        options=['age', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'Dalc', 'Walc', 'famrel'],
        index=1
    )
    
    # Y-axis selection  
    y_axis = st.selectbox(
        "Pilih variabel Y:",
        options=['G1', 'G2', 'G3'],
        index=2
    )
    
    # Color encoding
    color_by = st.selectbox(
        "Warna berdasarkan:",
        options=['sex', 'school', 'address', 'romantic', 'higher'],
        index=0
    )
    
    # Chart type
    chart_type_explore = st.selectbox(
        "Jenis visualisasi:",
        options=['Scatter Plot', 'Box Plot', 'Violin Plot', 'Bar Chart'],
        index=0
    )

with col2:
    st.subheader("Hasil Eksplorasi")
    
    if chart_type_explore == 'Scatter Plot':
        fig_explore = px.scatter(
            filtered_df, x=x_axis, y=y_axis, color=color_by,
            hover_data=['age', 'school', 'studytime'],
            title=f"{y_axis} vs {x_axis} (colored by {color_by})"
        )
    elif chart_type_explore == 'Box Plot':
        fig_explore = px.box(
            filtered_df, x=x_axis, y=y_axis, color=color_by,
            title=f"Distribusi {y_axis} berdasarkan {x_axis}"
        )
    elif chart_type_explore == 'Violin Plot':
        fig_explore = px.violin(
            filtered_df, x=x_axis, y=y_axis, color=color_by,
            box=True, title=f"Distribusi {y_axis} berdasarkan {x_axis}"
        )
    else:  # Bar Chart
        avg_data = filtered_df.groupby([x_axis, color_by])[y_axis].mean().reset_index()
        fig_explore = px.bar(
            avg_data, x=x_axis, y=y_axis, color=color_by,
            title=f"Rata-rata {y_axis} berdasarkan {x_axis}"
        )
    
    st.plotly_chart(fig_explore, use_container_width=True)

# ============ COMPARATIVE ANALYSIS ============
st.header(" Analisis Perbandingan")

st.markdown("Bandingkan performa antara grup yang berbeda:")

comparison_col1, comparison_col2 = st.columns(2)

with comparison_col1:
    st.subheader("Perbandingan Berdasarkan Gender")
    
    gender_comparison_grade = st.selectbox(
        "Pilih nilai untuk perbandingan gender:",
        options=['G1', 'G2', 'G3'],
        index=2,
        key="gender_comparison"
    )
    
    # Statistical comparison
    male_scores = filtered_df[filtered_df['sex'] == 'M'][gender_comparison_grade]
    female_scores = filtered_df[filtered_df['sex'] == 'F'][gender_comparison_grade]
    
    comparison_stats = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Male': [male_scores.mean(), male_scores.median(), male_scores.std(), 
                male_scores.min(), male_scores.max()],
        'Female': [female_scores.mean(), female_scores.median(), female_scores.std(),
                  female_scores.min(), female_scores.max()]
    })
    
    st.dataframe(comparison_stats.round(2))
    
    # Visual comparison
    fig_gender_comp = px.histogram(
        filtered_df, x=gender_comparison_grade, color='sex',
        barmode='overlay', opacity=0.7,
        title=f"Distribusi {gender_comparison_grade} berdasarkan Gender"
    )
    st.plotly_chart(fig_gender_comp, use_container_width=True)

with comparison_col2:
    st.subheader("Perbandingan Berdasarkan Sekolah")
    
    school_comparison_grade = st.selectbox(
        "Pilih nilai untuk perbandingan sekolah:",
        options=['G1', 'G2', 'G3'],
        index=2,
        key="school_comparison"
    )
    
    # School statistics
    school_stats = filtered_df.groupby('school')[school_comparison_grade].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).round(2)
    st.dataframe(school_stats)
    
    # Visual comparison
    fig_school_comp = px.box(
        filtered_df, x='school', y=school_comparison_grade,
        color='school', title=f"Distribusi {school_comparison_grade} per Sekolah"
    )
    st.plotly_chart(fig_school_comp, use_container_width=True)

# ============ TREND ANALYSIS ============
st.header(" Analisis Tren")

st.subheader("Tren Nilai dari G1 ke G3")

# Create trend analysis
trend_analysis = st.radio(
    "Pilih jenis analisis tren:",
    ["Tren Individual", "Tren Berdasarkan Grup"],
    horizontal=True
)

if trend_analysis == "Tren Individual":
    # Sample individual student trends
    sample_size = st.slider("Jumlah sample siswa:", min_value=5, max_value=50, value=20)
    
    sample_students = filtered_df.sample(n=min(sample_size, len(filtered_df)))
    trend_data = []
    
    for idx, row in sample_students.iterrows():
        for grade in ['G1', 'G2', 'G3']:
            trend_data.append({
                'Student_ID': f"Student_{idx}",
                'Grade_Period': grade,
                'Score': row[grade],
                'Sex': row['sex'],
                'School': row['school']
            })
    
    trend_df = pd.DataFrame(trend_data)
    
    fig_trend_individual = px.line(
        trend_df, x='Grade_Period', y='Score', color='Student_ID',
        title=f"Tren Nilai Individual ({sample_size} siswa sample)",
        facet_col='Sex'
    )
    fig_trend_individual.update_layout(showlegend=False)
    st.plotly_chart(fig_trend_individual, use_container_width=True)

elif trend_analysis == "Tren Berdasarkan Grup":
    # Group-based trends
    group_by = st.selectbox(
        "Grup berdasarkan:",
        options=['sex', 'school', 'address', 'higher'],
        key="trend_group"
    )
    
    group_trend_data = []
    for group in filtered_df[group_by].unique():
        group_data = filtered_df[filtered_df[group_by] == group]
        for grade in ['G1', 'G2', 'G3']:
            group_trend_data.append({
                'Group': group,
                'Grade_Period': grade,
                'Avg_Score': group_data[grade].mean(),
                'Count': len(group_data)
            })
    
    group_trend_df = pd.DataFrame(group_trend_data)
    
    fig_trend_group = px.line(
        group_trend_df, x='Grade_Period', y='Avg_Score', color='Group',
        markers=True, title=f"Tren Rata-rata Nilai berdasarkan {group_by}"
    )
    st.plotly_chart(fig_trend_group, use_container_width=True)


