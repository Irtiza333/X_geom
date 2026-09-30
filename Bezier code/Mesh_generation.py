#the mesh script needs to read in sample_blade{x1}.iges and generate a mesh for each of them by naming them as ORCA{x1}.cas and so on
# For the cases where the script fails to generate a mesh, it tries to other avaiable mesh scripts. 
# If all the mesh scripts fail, the code writes in a file called "mesh_script_failed.txt" which cad files failed to mesh.
# All the successful mesh files are moved into the "Mesh_files" folder.

import os


## CFD mesh generation ##
N_samples = 60
mesh_scripts = ['mesh_script.glf', 'mesh_script_2.glf', 'mesh_script_3.glf']

 
def Mesh_generation(x1):
    value = 0
    for mesh_script in mesh_scripts:
        cmd_pw = f'/opt/software/Pointwise/Pointwise2023.2/pointwise -b {mesh_script}'
        os.system(cmd_pw)
        # check if the mesh file exists meaning that the mesh script succeeded
        if os.path.exists(f'ORCA{x1}.cas'):
            print(f'Mesh script {mesh_script} succeeded')
            value = 1
            if not os.path.isdir('Mesh_files'):
                os.makedirs('Mesh_files', exist_ok=True)
            cmd_move = f'mv ORCA{x1}.cas Mesh_files/'
            os.system(cmd_move)
            # if mesh script succeeded, break the loop
            break
    if value == 0:
        with open('mesh_script_failed.txt', 'a') as f:
            f.write(f'all mesh scripts failed for {x1}\n')

for i in range(N_samples):
    x1 = i
    Mesh_generation(x1)
