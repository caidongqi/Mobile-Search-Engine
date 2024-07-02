for s in {1,5,10};do
    for q  in {1,2,5,10,20,30,40,50};do
        python e2e_harsmart_ground_truth.py --Q $q --S $s 
    done
done