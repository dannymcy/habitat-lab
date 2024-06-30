import pandas as pd
import os
import json


def update_motion_type(scene_data, objects_to_make_dynamic):
    for obj in scene_data['object_instances']:
        if obj['template_name'] in objects_to_make_dynamic:
            obj['motion_type'] = 'DYNAMIC'
            obj['auto_clamp_joint_limits'] = True
    return scene_data


def get_scene_files(directory):
    scene_files_list = []
    scene_full_list = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".scene_instance.json"):
            scene_files_list.append(filename)
            scene_full_list.append(os.path.join(directory, filename))
    
    return scene_files_list, scene_full_list



if __name__ == "__main__":
    # Load the CSV file
    csv_file_path = 'data/scene_datasets/hssd-hab/semantics/objects.csv'
    objects_df = pd.read_csv(csv_file_path)

    # Categories to consider for making objects dynamic
    unique_super_categories = objects_df['super_category'].unique()
    print()
    print(unique_super_categories)
    print()
    
    # dynamic_categories = ['trashcan', 'decor', 'lighting', 'seating_furniture', 'dining_ware', 'plant', 'electronics'
    #                       'animate_object', 'apparel', 'liquid_container', 'kitchen_ware', 'tray',
    #                       'small_kitchen_appliance', 'bathroom_accessory', 'gym_equipment', 'toy', 'wearable']
    # dynamic_categories = ['trashcan', 'decor', 'lighting', 'dining_ware', 'plant', 'electronics',
    #                       'animate_object', 'apparel', 'liquid_container', 'kitchen_ware', 'tray',
    #                       'small_kitchen_appliance', 'bathroom_accessory', 'gym_equipment', 'toy', 'wearable']
    dynamic_categories = []
    static_categories = ["storage_furniture", "support_furniture", "seating_furniture", "floor_covering", 
                         "sleeping_furniture", "bathroom_fixtures", "mirror",
                         "large_kitchen_appliance", "large_appliance", "kitchen_bathroom_fixture", 
                         "vehicle", "heating_cooling", "medium_kitchen_appliance", "display"]

    # List of objects to make dynamic by their IDs
    # TODO: NaN is an important categort, containing many objects. Need to resolve this.
    # objects_to_make_dynamic = objects_df[
    #     (objects_df['support'].isna() | (objects_df['support'] == "")) &
    #     (objects_df['super_category'].isin(dynamic_categories) | objects_df['super_category'].isna())
    # ]['id'].tolist()
    objects_to_make_dynamic = objects_df[
        (objects_df['support'].isna() | (objects_df['support'] == "")) &
        (objects_df['super_category'].isin(dynamic_categories))
    ]['id'].tolist()

    scene_dir = 'data/hab3_bench_assets/hab3-hssd/scenes_all_static'
    output_dir = 'data/hab3_bench_assets/hab3-hssd/scenes'
    scene_files_list, scene_full_list = get_scene_files(scene_dir)

    for i, scene_file in enumerate(scene_full_list):
        output_path = os.path.join(output_dir, scene_files_list[i])
        with open(scene_file, 'r') as f:
            scene_data = json.load(f)

        # Update the motion_type
        updated_scene_data = update_motion_type(scene_data, objects_to_make_dynamic)

        # Save the modified JSON file
        with open(output_path, 'w') as f:
            json.dump(updated_scene_data, f, indent=4)