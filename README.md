# PROJECT OVERVIEW

**Project Goal:** Estimate obesity levels using multiclass classification machine learning algorithms such as Decision Tree, Knn, Logistic Regression and multiclass methods like "One-vs-One" (OvO) ve "One-vs-Rest" (OvR).

**Input Data:**
Daily Living Habits: Eating patterns, physical activity levels, smoking, family history of overweight.
Demographic Features: Height, weight, age.

**Calculated Variables:**
* BMI (Body Mass Index)
* BMR (Basal Metabolic Rate)
* Ideal Weight
* Daily Calorie Intake
  
**Recommendations:**
Meal Recommendations: Based on DCI, diet type and cuisine preferences.
Alternative Meals: Provided using Content-Based Filtering with Nutritional Similarity algorithm.

# DATASETS

**INFORMATION ABOUT WEIGHT WISE DATASETS**

The datasets used are ObesityDataSet_raw_and_data_sinthetic.csv and All_Diets.csv from Kaggle.

**Obesity Dataset:** 

The dataset was obtained from Kaggle and was originally collected by the research team of Dr. Paulo Cortez and Prof. Ana Almeida from the University of Minho, Portugal.
Citation : P. Cortez and A. Almeida. "Predicting Obesity Type Based on Genetic and LifeStyle Factors." In Proceedings of the 5th International Workshop on Knowledge Discovery in Databases. Porto, Portugal, 2005.

* https://www.kaggle.com/code/mpwolke/obesity-levels-life-style

**All Diets Dataset:**

The file All_Diets.csv contains recipes from different diets and cuisines, all with the aim of providing healthy and nutritious meal options. Collaborator is The Devastator (Owner).

* https://www.kaggle.com/datasets/thedevastator/healthy-diet-recipes-a-comprehensive-dataset/data

**Columns of “ObesityDataSet_raw_and_data_sinthetic.csv” Dataset**

* Gender: Male or Female
* Age: Age of the individual in years
* Height: Height of the individual in meters
* Weight: Weight of the individual in kilograms
* Family_history_with_overweight: Has the individual a family history of overweight or obesity? Yes or No
*	FAVC: Does the individual consume high caloric food frequently? Yes or No
* FCVC: How often does the individual consume vegetables? 1 (never) to 3 (always)
* NCP: How many main meals does the individual have daily? 1 to 3
* CAEC: Does the individual monitor the calories they eat? Sometimes, Frequently, Always or No
* SMOKE: Does the individual smoke? Yes or No
* CH2O: How much water does the individual drink daily? 1 (less than a liter), 2 (1 to 2 liters), or 3 (more than 2 liters)
* SCC: Does the individual monitor the calories they burn? Yes or No
* FAF: How often does the individual engage in physical activity? 0 (never) to 3 (always)
* TUE: How many hours does the individual spend sitting on a typical day? 0 (less than an hour), 1 (1 to 2 hours), or 2 (more than 2 hours)
* CALC: Does the individual take extra calories? Always, Sometimes or No
* MTRANS: Transportation method used by the individual: Automobile, Bike, Motorbike, Public Transportation or Walking
* NObeyesdad: Obesity level of the individual, classified into: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II or Obesity_Type_III

**Columns of “All_Diets.csv” Dataset**

* Diet_type: The type of diet the recipe is for.
* Recipe_name: The name of the recipe.
* Cuisine_type: The cuisine the recipe is from.
* Protein(g): The amount of protein in grams.
* Carbs(g): The amount of carbs in grams.
* Fat(g): The amount of fat in grams.
* Extraction_day: The day the recipe was extracted.
