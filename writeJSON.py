# Author: RuiXin Gong

import project_helper
import json

id_list, cuisin_list, ingredients_list = project_helper.read_data("train.json")
clean_ingredients_list, deleted_list = project_helper.clean_fix_ingredients(
    ingredients_list)
compress_clean_ingredients_list = project_helper.compress_ingredients(
    clean_ingredients_list)

write_Json = []
for id, cuisin, ingredients in zip(id_list, cuisin_list, compress_clean_ingredients_list):
    dic = {
        "id": id,
        "cuisine": cuisin,
        "ingredients": ingredients,
    }
    write_Json.append(dic)
with open("train_postprecessing.json", "w") as outfile:
    json_object = json.dumps(write_Json)
    outfile.write(json_object)
