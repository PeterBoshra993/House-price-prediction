from flask import Flask, request, jsonify, render_template
import joblib
# import xgboost as xgb



## step to make my local computer as a server
app = Flask(__name__) ## __name__ = my current file name "price_pred_house.py" and make it my server
# model_xgb_2 = xgb.Booster()
# model = model_xgb_2.load_model("model.h5")
# model = model_xgb_2.load_model("model using xgboostgridsearch.h5")
model = joblib.load('LR model 90 train 83 test.h5')
scaler = joblib.load('scaler.h5')
def extract_client_from_request(req_data):
    client = []
    client.append(req_data['UnderGrad Bath Type'] == 'BsmtFullBath')
    client.append(req_data['UnderGrad Bath Type'] == 'BsmtHalfBath')
    
    client.append(req_data['AboveGrad Bath Type'] == 'FullBath')
    client.append(req_data['AboveGrad Bath Type'] == 'HalfBath')
    
    client.append(req_data['BedroomAbvGr']) # number of bedrooms above grad
    client.append(req_data['KitchenAbvGr']) # number of kitchens above grad
    client.append(req_data['total_rooms_above_grade']) # number of total rooms above grad
    client.append(req_data['Fireplaces']) # number of fireplaces
    client.append(req_data['garage_car_capacity']) # number of cars can enter to the garage
    client.append(req_data['other_features_values']) # price of other features of house
    client.append(req_data['year_diff']) # how many years from the construction of house and remod year

    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_1-1/2 STORY FINISHED ALL AGES') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_1-STORY 1945 & OLDER') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_1-STORY 1946 & NEWER ALL STYLES') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_1-STORY PUD (Planned Unit Development) - 1946 & NEWER') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_1-STORY W/FINISHED ATTIC ALL AGES') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_2 FAMILY CONVERSION - ALL STYLES AND AGES') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_2-1/2 STORY ALL AGES') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_2-STORY 1945 & OLDER') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_2-STORY 1946 & NEWER') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_2-STORY PUD - 1946 & NEWER') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_DUPLEX - ALL STYLES AND AGES') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_PUD - MULTILEVEL - INCL SPLIT LEV/FOYER') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_SPLIT FOYER') # Dwelling involved type
    client.append(req_data['Dwelling Involved'] == 'dwelling_involved_type_SPLIT OR MULTI-LEVEL') # Dwelling involved type

    
    client.append(req_data['General zoning classification'] == 'general_zoning_classification_RL') # General zoning classification RL / RM
    client.append(req_data['General zoning classification'] == 'general_zoning_classification_RM') # General zoning classification RL / RM
    
    client.append(req_data['type_of_road_Pave']) # 0, 1

    client.append(req_data['property_general_shape'] == 'property_general_shape_Other') # 1 if other or 0 if not 
    client.append(req_data['property_general_shape'] == 'property_general_shape_Reg') # 1 if Regular or 0 if not


    client.append(req_data['property_Flatness_Other']) # 0 if flat or 1 if other
    client.append(req_data['utilities_types_NoSewa']) # 1 or 0 if other
    client.append(req_data['LotConfig_Other']) # 0 or 1 if other
    client.append(req_data['LandSlope_Other']) # 0 or 1 if other



    client.append(req_data['neighborhood'] == 'Neighborhood_Blueste')
    client.append(req_data['neighborhood'] == 'Neighborhood_BrDale')
    client.append(req_data['neighborhood'] == 'Neighborhood_BrkSide')
    client.append(req_data['neighborhood'] == 'Neighborhood_ClearCr')
    client.append(req_data['neighborhood'] == 'Neighborhood_CollgCr')
    client.append(req_data['neighborhood'] == 'Neighborhood_Crawfor')
    client.append(req_data['neighborhood'] == 'Neighborhood_Edwards')
    client.append(req_data['neighborhood'] == 'Neighborhood_Gilbert')
    client.append(req_data['neighborhood'] == 'Neighborhood_IDOTRR')
    client.append(req_data['neighborhood'] == 'Neighborhood_MeadowV')
    client.append(req_data['neighborhood'] == 'Neighborhood_Mitchel')
    client.append(req_data['neighborhood'] == 'Neighborhood_NAmes')
    client.append(req_data['neighborhood'] == 'Neighborhood_NPkVill')    
    client.append(req_data['neighborhood'] == 'Neighborhood_NWAmes')
    client.append(req_data['neighborhood'] == 'Neighborhood_NoRidge')
    client.append(req_data['neighborhood'] == 'Neighborhood_NridgHt')
    client.append(req_data['neighborhood'] == 'Neighborhood_OldTown')
    client.append(req_data['neighborhood'] == 'Neighborhood_SWISU')
    client.append(req_data['neighborhood'] == 'Neighborhood_Sawyer')
    client.append(req_data['neighborhood'] == 'Neighborhood_SawyerW')
    client.append(req_data['neighborhood'] == 'Neighborhood_Somerst')
    client.append(req_data['neighborhood'] == 'Neighborhood_StoneBr')
    client.append(req_data['neighborhood'] == 'Neighborhood_Timber')
    client.append(req_data['neighborhood'] == 'Neighborhood_Veenker')

    client.append(req_data['dwelling_type_Other']) # 1 if other or 0 if not

    client.append(req_data['House Style'] == 'HouseStyle_1Story')
    client.append(req_data['House Style'] == 'HouseStyle_2Story')
    client.append(req_data['House Style'] == 'HouseStyle_Other')

    client.append(req_data['Overall Quality'] == 'OverallQual_Very Excellent')
    client.append(req_data['Overall Quality'] == 'OverallQual_above average')
    client.append(req_data['Overall Quality'] == 'OverallQual_average')
    client.append(req_data['Overall Quality'] == 'OverallQual_below average')
    client.append(req_data['Overall Quality'] == 'OverallQual_fair')
    client.append(req_data['Overall Quality'] == 'OverallQual_good')
    client.append(req_data['Overall Quality'] == 'OverallQual_poor')
    client.append(req_data['Overall Quality'] == 'OverallQual_very good')
    
    client.append(req_data['Overall Condition'] == 'OverallCond_above average')
    client.append(req_data['Overall Condition'] == 'OverallCond_average')
    client.append(req_data['Overall Condition'] == 'OverallCond_below average')
    client.append(req_data['Overall Condition'] == 'OverallCond_fair')
    client.append(req_data['Overall Condition'] == 'OverallCond_good')
    client.append(req_data['Overall Condition'] == 'OverallCond_poor')
    client.append(req_data['Overall Condition'] == 'OverallCond_very good')
    
    client.append(req_data['Roof Style'] == 'RoofStyle_Hip')
    client.append(req_data['Roof Style'] == 'RoofStyle_Other')

    client.append(req_data['roof_material_Other']) # 0,1

    client.append(req_data['exterior_covering_1'] == 'exterior_covering_1_MetalSd')
    client.append(req_data['exterior_covering_1'] == 'exterior_covering_1_Other')
    client.append(req_data['exterior_covering_1'] == 'exterior_covering_1_Plywood')
    client.append(req_data['exterior_covering_1'] == 'exterior_covering_1_VinylSd')
    client.append(req_data['exterior_covering_1'] == 'exterior_covering_1_Wd Sdng')



    client.append(req_data['exterior_covering_2'] == 'exterior_covering_2_MetalSd')
    client.append(req_data['exterior_covering_2'] == 'exterior_covering_2_Other')
    client.append(req_data['exterior_covering_2'] == 'exterior_covering_2_Plywood')
    client.append(req_data['exterior_covering_2'] == 'exterior_covering_2_VinylSd')
    client.append(req_data['exterior_covering_2'] == 'exterior_covering_2_Wd Sdng')


    client.append(req_data['masonry_veneer_type'] == 'masonry_veneer_type_None')
    client.append(req_data['masonry_veneer_type'] == 'masonry_veneer_type_Other')


    client.append(req_data['Exterior Quality'] == 'ExterQual_Other')
    client.append(req_data['Exterior Quality'] == 'ExterQual_TA')



    client.append(req_data['Exterior Condition'] == 'ExterQual_Other')
    client.append(req_data['Exterior Condition'] == 'ExterQual_TA')


    client.append(req_data['Basement Quality'] == 'BsmtQual_Other')
    client.append(req_data['Basement Quality'] == 'BsmtQual_TA')



    client.append(req_data['BsmtCond_TA']) # 0, 1

    client.append(req_data['Basement Exposure'] == 'BsmtExposure_Gd')
    client.append(req_data['Basement Exposure'] == 'BsmtExposure_Mn')
    client.append(req_data['Basement Exposure'] == 'BsmtExposure_No')


    client.append(req_data['Basement Finish Type 1'] == 'BsmtFinType1_BLQ')
    client.append(req_data['Basement Finish Type 1'] == 'BsmtFinType1_GLQ')
    client.append(req_data['Basement Finish Type 1'] == 'BsmtFinType1_LwQ')
    client.append(req_data['Basement Finish Type 1'] == 'BsmtFinType1_Rec')
    client.append(req_data['Basement Finish Type 1'] == 'BsmtFinType1_Unf')

    client.append(req_data['Basement Finish Type 2'] == 'BsmtFinType2_BLQ')
    client.append(req_data['Basement Finish Type 2'] == 'BsmtFinType2_GLQ')
    client.append(req_data['Basement Finish Type 2'] == 'BsmtFinType2_LwQ')
    client.append(req_data['Basement Finish Type 2'] == 'BsmtFinType2_Rec')
    client.append(req_data['Basement Finish Type 2'] == 'BsmtFinType2_Unf')


    client.append(req_data['Heating_Other']) # 0,1

    client.append(req_data['Heating Quality Control'] == 'HeatingQC_Gd')
    client.append(req_data['Heating Quality Control'] == 'HeatingQC_TA')
    client.append(req_data['Heating Quality Control'] == 'HeatingQC_Other')


    client.append(req_data['CentralAir_Y']) # 0,1
    client.append(req_data['electrical_system_SBrkr']) # 0,1

    client.append(req_data['Kitchen Quality'] == 'KitchenQual_Other')
    client.append(req_data['Kitchen Quality'] == 'KitchenQual_TA')
    
    
    client.append(req_data['Functional_Typ']) #0,1


    client.append(req_data['Garage Type'] == 'GarageType_Detchd')
    client.append(req_data['Garage Type'] == 'GarageType_Other')


    client.append(req_data['Garage interior finish'] == 'interior_finish_garage_RFn')
    client.append(req_data['Garage interior finish'] == 'interior_finish_garage_Unf')

    client.append(req_data['GarageQual_TA']) #0,1
    client.append(req_data['GarageCond_TA']) #0,1

    
    client.append(req_data['Paved Drive'] == 'PavedDrive_P')
    client.append(req_data['Paved Drive'] == 'PavedDrive_Y')




    client.append(req_data['Month Sold'] == 'MoSold_Aug')
    client.append(req_data['Month Sold'] == 'MoSold_Dec')
    client.append(req_data['Month Sold'] == 'MoSold_Feb')
    client.append(req_data['Month Sold'] == 'MoSold_Jan')
    client.append(req_data['Month Sold'] == 'MoSold_Jul')
    client.append(req_data['Month Sold'] == 'MoSold_Jun')
    client.append(req_data['Month Sold'] == 'MoSold_Mar')
    client.append(req_data['Month Sold'] == 'MoSold_May')
    client.append(req_data['Month Sold'] == 'MoSold_Nov')
    client.append(req_data['Month Sold'] == 'MoSold_Oct')
    client.append(req_data['Month Sold'] == 'MoSold_Sep')


    client.append(req_data['Year Sold'] == 'YrSold_2007')
    client.append(req_data['Year Sold'] == 'YrSold_2008')
    client.append(req_data['Year Sold'] == 'YrSold_2009')
    client.append(req_data['Year Sold'] == 'YrSold_2010')    

    client.append(req_data['SaleType_WD']) # 0,1
    client.append(req_data['SaleCondition_Other']) # 0,1
    client.append(req_data['Is_diff_difference']) # 0,1
    client.append(req_data['Condition_all_Other']) # 0,1

    client.append(req_data['BasementExpo'] == 'Bsmt Exposure_No')
    client.append(req_data['BasementExpo'] == 'Bsmt Exposure_Other')


    client.append(req_data['Pool_exist_exist']) # 0,1
    client.append(req_data['2ndFlr_exist_exist']) # 0,1
    client.append(req_data['WoodDeck_exist_exist']) # 0,1
    client.append(req_data['OpenPorch_exist_exist']) # 0,1
    client.append(req_data['masonry_veneer_exist_exist']) # 0,1
    client.append(req_data["Low_Quality_areas_existance_there's low quality"]) # 0,1


    client.append(req_data['Quantile Ranges of Areas'] == 'quantile_ranges_of_areas_middle')
    client.append(req_data['Quantile Ranges of Areas'] == 'quantile_ranges_of_areas_top')


    client.append(req_data['Percentage of Basement Finished Areas'] == 'quantile_ranges_of_basement_finished_areas_nearly 50% finished')
    client.append(req_data['Percentage of Basement Finished Areas'] == 'quantile_ranges_of_basement_finished_areas_mostly finished')



    client.append(req_data['three_enteries_exist_exist']) # 0,1
    client.append(req_data['Open_Porch_exist_exist']) # 0,1
    client.append(req_data['Screen_Porch_exist_exist']) # 0,1
    client.append(req_data['Enclosed Porch exist_exist']) # 0,1
    return client
@app.route('/predict', methods = ['POST'])

def SalePrice():
    client = extract_client_from_request(request.json)
    client = scaler.transform([client])
    prediction = model.predict(client)[0]
    return jsonify({'House Sale Price in thousands': prediction/1000})


if __name__ == '__main__':
    
    app.run(debug = True)