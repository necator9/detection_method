#!/bin/bash

declare -a arr=("woman_1" "thin_man" "short_man" "short_group" "pair_1" "pair_2"
                "tall_group_1" "tall_group_2" "cyclist_1" "disp_car")


for i in "${arr[@]}"
do
   python get_features.py "$i" &
done

wait
echo "Complete!"

