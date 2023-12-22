import subprocess
import os
import shutil
import platform

chainlit_command = 'chainlit run chat_pdf.py '

print("Ur running on: ", platform.system())

# Run the chainnlit process
try:
    # Run the chainlit process
    chainlit_process = subprocess.run(chainlit_command, shell=True, check=True)
    if platform.system() == "Darwin" or platform.system() == "Linux":
        database_path = "./database"
        shutil.rmtree(database_path)
        print("Deleting the temporary database")
        

except subprocess.CalledProcessError as e:
    print("The chainlit process has been suddenly been closed by user!")

    database_path = "./database"
    if os.path.exists(database_path):
        shutil.rmtree(database_path)
        print("Deleting the temporary database")
    else:
        print("Path not exist")