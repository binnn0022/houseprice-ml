import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("my_model.pkl")

mszoning_options = ['C (all)', 'FV', 'RH', 'RL', 'RM']
lotconfig_options = ['Corner', 'CulDSac', 'FR2', 'FR3', 'Inside']
bldgtype_options = ['1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE']
exterior_options = [
    'AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock',
    'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Plywood',
    'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing'
]

st.title("House Features Selection")

ms_subclass = st.number_input("MSSubClass", min_value=0, value=60, max_value=190)
lot_area = st.number_input("Lot Area (sq ft)", min_value=0, value=5000, max_value=215245)
overall_cond = st.number_input("Overall Condition (1-10)", min_value=1, max_value=10, value=9)
year_built = st.number_input("Year Built", min_value=1872, max_value=2023, value=2010)
year_remod_add = st.number_input("Year Remodeled/Additions", min_value=1950, max_value=2010, value=2005)
bsmt_fin_sf2 = st.number_input("Basement Finished Area 2 (sq ft)", min_value=0, value=0, max_value=1526)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, value=1000, max_value=6110)

selected_mszoning = st.selectbox("Select MSZoning", mszoning_options)

st.info("""
- **MSZoning**: Classification of the property's zoning.
  - **RL**: Residential Low Density
  - **RM**: Residential Medium Density
  - **C (all)**: Commercial (all types)
  - **FV**: Floating Village
  - **RH**: Residential High Density
""")

selected_lotconfig = st.selectbox("Select Lot Config", lotconfig_options)

st.info("""
- **LotConfig**: Configuration of the lot.
  - **Inside**: Standard lot within a block.
  - **FR2**: Lot with a front yard depth of 2 feet.
  - **Corner**: Lot at the intersection of two streets.
  - **CulDSac**: Lot at the end of a cul-de-sac.
  - **FR3**: Lot with a front yard depth of 3 feet.
""")

selected_bldgtype = st.selectbox("Select Building Type", bldgtype_options)


st.info("""
- **Building Type**: Type of dwelling on the property.
  - **1Fam**: Single-family dwelling.
  - **2fmCon**: Two-family conversion dwelling.
  - **Duplex**: Two-family dwelling.
  - **TwnhsE**: End unit townhouse.
  - **Twnhs**: Interior unit townhouse.
""")


selected_exterior = st.selectbox("Select Exterior", exterior_options)

st.info("""
- **Exterior**: Material used for the exterior of the house.
  - **VinylSd**: Vinyl siding.
  - **MetalSd**: Metal siding.
  - **Wd Sdng**: Wood siding.
  - **HdBoard**: Hardboard siding.
  - **BrkFace**: Brick face.
  - **WdShing**: Wood shingle.
  - **CemntBd**: Cement board.
  - **Plywood**: Plywood siding.
  - **AsbShng**: Asbestos siding.
  - **Stucco**: Stucco finish.
  - **BrkComm**: Brick commercial.
  - **AsphShn**: Asphalt shingle.
  - **Stone**: Stone exterior.
  - **ImStucc**: Immitation stucco.
  - **CBlock**: Cinder block.
""")

predictbutton = st.button("Predict!")

if predictbutton:
    st.balloons()

    data = {
        "MSSubClass": [ms_subclass],
        "LotArea": [lot_area],
        "OverallCond": [overall_cond],
        "YearBuilt": [year_built],
        "YearRemodAdd": [year_remod_add],
        "BsmtFinSF2": [bsmt_fin_sf2],
        "TotalBsmtSF": [total_bsmt_sf]
    }

    for option in mszoning_options:
        data[f"MSZoning_{option}"] = [1 if option == selected_mszoning else 0]
    
    for option in lotconfig_options:
        data[f"LotConfig_{option}"] = [1 if option == selected_lotconfig else 0]
    
    for option in bldgtype_options:
        data[f"BldgType_{option}"] = [1 if option == selected_bldgtype else 0]
    
    for option in exterior_options:
        data[f"Exterior1st_{option}"] = [1 if option == selected_exterior else 0]

    df = pd.DataFrame(data)

    all_columns = [
        "MSSubClass", "LotArea", "OverallCond", "YearBuilt",
        "YearRemodAdd", "BsmtFinSF2", "TotalBsmtSF"
    ] + [f"MSZoning_{option}" for option in mszoning_options] + \
      [f"LotConfig_{option}" for option in lotconfig_options] + \
      [f"BldgType_{option}" for option in bldgtype_options] + \
      [f"Exterior1st_{option}" for option in exterior_options]

    for column in all_columns:
        if column not in df.columns:
            df[column] = 0  

    df = df[all_columns]

    df.reset_index(drop=True, inplace=True)
    
    prediction = model.predict(df)  

    st.write(f"Price prediction is ${prediction[0]:,.2f}")
else:
    st.write("Please use the predict button after entering values")