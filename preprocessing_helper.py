# Author: RuiXin Gong

import json
import re
import unidecode


def read_data(file_name):
    with open(file_name) as f:
        data = json.load(f)
    id_list = []
    cuisin_list = []
    ingredients_list = []
    for element in data:
        id_list.append(element['id'])
        cuisin_list.append(element['cuisine'].lower())
        temp = []
        for e in element['ingredients']:
            no_space_list = e.split(" ")
            for i in no_space_list:
                if i:
                    temp.append(i.lower())
        ingredients_list.append(temp)
    return id_list, cuisin_list, ingredients_list


def clean_fix_ingredients(ingredients_list):
    result = []
    deleted_list = []
    for ingredients in ingredients_list:
        temp = []
        for element in ingredients:
            new_str = re.sub('[^a-z]', '', unidecode.unidecode(element))
            if (not new_str) or (len(new_str) <= 2):
                if element not in deleted_list:
                    deleted_list.append(element)
            else:
                temp.append(new_str)
        result.append(temp)
        if not temp:
            print("Warning! A ingredients_list becomes empty!")
    return result, deleted_list


def compress_ingredients(ingredients_list):
    result = []
    for ingredients in ingredients_list:
        result.append(" ".join(ingredients))
    return result
