#!/bin/bash

declare -a arr=("woman_1" "thin_man" "short_man" "short_group" "pair_1" "pair_2"
                "tall_group_1" "tall_group_2" "cyclist_1" "disp_car" "car_1")

if [ ! -d "$1" ]
then
    mkdir "$1"
fi

for i in "${arr[@]}"
do
   python get_features.py "$i" "$1" &
done

wait
echo "Complete!"

