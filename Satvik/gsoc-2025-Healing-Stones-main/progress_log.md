# Day 1: 
## What I did today:
1. Read and understood the GSoC Healing Stones test task under the HumanAI organization.
2. Downloaded the 3D dataset (17 .PLY fragments) from the Box.com link.
3. Installed required dependencies:
    open3d, numpy, matplotlib
4. Created a clean and organized project structure
5. Initialized a GitHub repo for version control.
6. Learned how to handle large files in Git using: .gitignore to exclude .PLY and .OBJ
7. Visualized a single 3D fragment using open3d.

   ![image](https://github.com/user-attachments/assets/fe102630-94df-461a-aa8d-edf4d714c6d5)

## Key Learning:
1. 3D visualization using open3d.draw_geometries()
2. Case sensitivity in file extensions can break visualization if not handled carefully
3. GitHub doesn’t support large 3D files well, better to externalize with Drive

# Day 2:
## What I did today:
1. compare_downsample.py:<br>
   a. Created a script to visually compare original vs downsampled point clouds.<br>
   b. Understood how voxel downsampling reduces the number of points while preserving the shape.<br>
   c. Visualized the result in color-coded 3D (blue = original, red = downsampled).<br>
   Result:<br>
   Original point count: 11860704<br>
   Downsampled point count: 880549<br>

3. icp_align.py:<br>
   a. Wrote a script to perform basic ICP alignment between two selected fragments using point-to-point ICP.<br>
   b. How ICP computes transformation matrices.<br>
   c. Visualizing results with aligned and target fragments.
   ![image](https://github.com/user-attachments/assets/e0b57d12-cb1a-4f41-b1b4-5653eb90c45d)<br><br>


5. icp_align2.py:<br>
   a. Extended ICP logic to compare point-to-point vs point-to-plane ICP.<br>
   b. Evaluated result using fitness and RMSE.<br>
   c. Observed that point-to-plane gave slightly better fitness and more detailed surface matching.<br>
   ![image](https://github.com/user-attachments/assets/7a39636c-e957-49a0-a001-3cf21e90352c)<br><br>


7. pairwise_icp_log.py:<br>
   a. Developed a script to automatically align multiple fragment pairs using point-to-plane ICP only.<br>
   b. Selected the first 5 fragments, and compared all unique combinations (10 pairs total).<br>
   c. Logged the results (Source, Target, Fitness, RMSE) into a CSV file: icp_results.csv<br>
   d. Optimized loop to avoid self-pairs and reversed duplicates by comparing only i < j.<br>
   ![image](https://github.com/user-attachments/assets/bf0a8e22-a65d-44a6-a14f-7f145a96a547)<br>

## Key Learning:
1. ICP performance varies significantly by fragment pair.
2. Point-to-plane is better suited for cultural heritage fragments.
3. Automating and logging results gives clear insight into matching quality.

# Day 3:
## What I did today:

1. analyse_icp_results.py:<br>
   a. Automatically sort and print top fragment pairs based on fitness.<br>
   b. Observed that some top-scoring pairs were actually duplicates (identified by identical point counts and identity transformation matrix).<br>
   c. Confirmed working ICP alignment for meaningful pairs.<br>

2. global_assembly.py:<br>
  a. Reads sorted_icp_results.csv<br>
  b. Selects a base fragment (most connections in top matches).<br>
  c. Iteratively aligns and merges fragments using point-to-plane ICP.<br>
  d. Maintains a global assembly with cumulative transformations.<br>
![image](https://github.com/user-attachments/assets/ee0a775e-153a-42cc-bc0d-a869a6484270)

# Day 4:
## What I did today:

### Clean_colorise_aseembly.py:
a. Optimized point cloud.<br>
b. Downsampled for clarity and reduced size.<br>
c. Statistical outliers removed.<br>
d. Added random colors across the fragments for visual inspection.<br>
![image](https://github.com/user-attachments/assets/390b367e-050d-42f5-882a-56351cdb43df)



