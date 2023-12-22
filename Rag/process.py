import subprocess
import os
import shutil

chainlit_command = 'chainlit run chat_pdf.py '


# Run the chainnlit process
try:
    # Run the chainlit process
    chainlit_process = subprocess.run(chainlit_command, shell=True, check=True)
    print("The chainlit process has been executed and closed")
except subprocess.CalledProcessError as e:
    print("The chainlit process has been executed and closed")

    database_path = "./database"

    if os.path.exists(database_path):
        shutil.rmtree(database_path)
        print("Deleting the temporary database")