import os
import logging
import json
import yaml
import tempfile
import glob
import zipfile
import sendgrid

import azure.functions as func

from io import BytesIO
from basic_inference_function import setup_inference, do_inference
from pathlib import Path
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from sendgrid.helpers.mail import Mail, Email, To, Content

import logging
logging.Logger.root.level = 10

logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)

app = func.FunctionApp()

def read_dicom_file(file_path: str) -> bytes:
    with open(file_path, 'rb') as file:
        dicom_data = file.read()
    return dicom_data

def upload_to_blob(container_name: str, blob_name: str, data: bytes, connection_string: str) -> None:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    blob_client.upload_blob(data, overwrite=True)

def upload_zip_to_blob(zip_buffer, container_name, blob_name, connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_container_client(container=container_name).get_blob_client(blob_name)
    # Upload the zip file
    blob_client.upload_blob(zip_buffer, blob_type="BlockBlob", overwrite=True)

def read_from_blob(container_name: str, blob_name: str, connection_string: str):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_content = blob_client.download_blob().readall()

    return blob_content

def save_blob_as_tmp(container_name: str, blob_name: str, connection_string: str, base_path: str) -> None:
    blob_content = read_from_blob(container_name, blob_name, connection_string)

    temp_file_path = Path(base_path) / blob_name # blob name is the full hierarchical structure path name,a nd we rebase this to the tmp base dir
    temp_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(blob_content)

    return temp_file_path

def create_zip_archive(temp_files):
    # Create a buffer to hold the zip file in memory
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for temp_file in temp_files:
            # Determine the archive name for the file
            archive_name = os.path.basename(temp_file)
            # Add the file to the zip, using its base name as the archive name
            zip_file.write(temp_file, arcname=archive_name)
    
    # Seek to the start of the BytesIO buffer
    zip_buffer.seek(0)
    return zip_buffer


def verify_files(blob_service_client, container_name, file_paths):
    # Check if all files in the manifest exist in the blob storage
    all_files_exist = True
    missing_files = []

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for fp in tqdm(file_paths, desc='checking if all files in the manifest are present in the blob storage'):
        if fp is None:
            continue

        blob_client = blob_service_client.get_blob_client(container=container_name, blob=fp)
        
        # Check if the blob exists
        if not blob_client.exists():
            all_files_exist = False
            missing_files.append(fp)

    if not all_files_exist:
        logging.error(f"The following files are missing from the blob storage: {missing_files}")
        return False
    
    logging.debug('all files in manifest are uploaded')
    return True


def send_completion_email(container_name, blob_name, download_url, recipient_email, expiry_hours=72):

    # connection_string = "your_storage_account_connection_string"
    # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    # sas_token = generate_blob_sas(account_name=account_name,
    #                               container_name=blob_container_name,
    #                               blob_name=blob_name,
    #                               account_key=account_key,
    #                               permission=BlobSasPermissions(read=True),
    #                               expiry=datetime.utcnow() + timedelta(hours=24))  # Link expires in 24 hours

    # download_url = f"https://{account_name}.blob.core.windows.net/{blob_container_name}/{blob_name}?{sas_token}"

    # Create the email content
    message = f"""\
        Your job is complete. You can download the results at the following link:

        {download_url}

        The link will expire in {expiry_hours} hours."""

    subject = f'Your Data Processing Job is Complete'
    
    send_email(recipient_email, subject, message)

def send_email(recipient_email, subject, message):
    sg = sendgrid.SendGridAPIClient(api_key=os.environ['SendGridToken'])
    from_email = Email("bjorn.ahlgren@elekta.com")  # Your SendGrid account email or verified sender
    to_email = To(recipient_email)
    content = Content("text/plain", message)
    mail = Mail(from_email, to_email, subject, content)
    response = sg.client.mail.send.post(request_body=mail.get())

    logging.info(response.status_code)
    logging.info(response.body)
    logging.info(response.headers)

def generate_sas_token(container_name, blob_name, account_name, account_key, expiry_hours=72):
    # account_name = os.environ['AzureWebJobsStorageName']
    # account_key = os.environ['AzureWebJobsStorageKey']
    sas_token = generate_blob_sas(account_name=account_name,
                                  container_name=container_name,
                                  blob_name=blob_name,
                                  account_key=account_key,
                                  permission=BlobSasPermissions(read=True),
                                  expiry=datetime.utcnow() + timedelta(hours=expiry_hours))
    return f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"

def flatten_list(nested_list):
    """
    Flatten an arbitrarily deeply nested list.

    Args:
        nested_list: The nested list to be flattened.

    Returns:
        The flattened list.
    """
    flattened_list = []
    for element in nested_list:
        if isinstance(element, list):
            flattened_list.extend(flatten_list(element))
        else:
            flattened_list.append(element)
    return flattened_list

def delete_uploaded_files(container_name, file_paths, connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container=container_name)
    file_paths = flatten_list(file_paths)
    for file_path in file_paths:
        try:
            blob_client = container_client.get_blob_client(blob=file_path)
            blob_client.delete_blob()
            logging.info(f"Deleted {file_path}")
        except Exception as e:
            logging.info(f"Error deleting {file_path}: {e}")

def ensure_unix_path_separators(path: str) -> str:
    path = path.replace('\\', '/')
    path = os.path.normpath(path)  # Normalize the path
    return path

############################################################
    
@app.blob_trigger(arg_name="inputblob", path="input-dicoms/manifest_{name}.yml", # name is treated as PatientID
                               connection="AzureWebJobsStorage") 
@app.queue_output(arg_name="msg", queue_name="process-queue", connection="AzureWebJobsStorage")
def blob_trigger(inputblob: func.InputStream, msg: func.Out[str]):
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {inputblob.name}"
                f"Blob Size: {inputblob.length} bytes")
    
    # read manifest file as a yml file
    stream = inputblob.read().decode("utf-8")
    manifest = yaml.safe_load(stream)

    # check that all files have been uploaded before creating a queue message
    connection_string = os.getenv('AzureWebJobsStorage')
    account_name = os.getenv('AzureWebJobsStorageName')
    account_key = os.getenv('AzureWebJobsStorageKey')
    manifest['connection_string'] = connection_string
    manifest['account_name'] = account_name
    manifest['account_key'] = account_key

    # Ensure UNIX-style path separators
    if 'dicom_folder' in manifest.keys():
        manifest['dicom_folder'] = ensure_unix_path_separators(manifest['dicom_folder'])
    if 'dicom_files' in manifest.keys():
        manifest['dicom_files'] = [ensure_unix_path_separators(fp) for fp in manifest['dicom_files']]
    if 'nifti_file' in manifest.keys():
        manifest['nifti_file'] = ensure_unix_path_separators(manifest['nifti_file'])
    if 'rtss' in manifest.keys():
        if manifest['rtss'] is not None:
            manifest['rtss'] = ensure_unix_path_separators(manifest['rtss'])

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = 'input-dicoms'

    file_paths = manifest['dicom_files']
    if not verify_files(blob_service_client, container_name, file_paths):
        return
    
    if manifest['rtss'] is not None:
        logging.info('rtss included in manifest')
        if not verify_files(blob_service_client, container_name, manifest['rtss']):
            return

    # Prepare the message for the queue
    queue_message = json.dumps(manifest)

    msg.set(queue_message)

############################################################

# @app.blob_trigger(arg_name="inputblob", path="input-dicoms/manifest_{name}.yml", # name is treated as PatientID
#                                connection="AzureWebJobsStorage_PAH") 
# @app.queue_output(arg_name="msg", queue_name="process-queue", connection="AzureWebJobsStorage") 
# def blob_trigger_PAH(inputblob: func.InputStream, msg: func.Out[str]):
#     logging.info(f"Python blob trigger function processed blob"
#                 f"Name: {inputblob.name}"
#                 f"Blob Size: {inputblob.length} bytes")
    
#     # read manifest file as a yml file
#     stream = inputblob.read().decode("utf-8")
#     manifest = yaml.safe_load(stream)

#     # check that all files have been uploaded before creating a queue message
#     connection_string = os.getenv('AzureWebJobsStorage_PAH')
#     account_name = os.getenv('AzureWebJobsStorageName_PAH')
#     account_key = os.getenv('AzureWebJobsStorageKey_PAH')
#     manifest['connection_string'] = connection_string
#     manifest['account_name'] = account_name
#     manifest['account_key'] = account_key

#     # Ensure UNIX-style path separators
#     if 'dicom_folder' in manifest.keys():
#         manifest['dicom_folder'] = ensure_unix_path_separators(manifest['dicom_folder'])
#     if 'dicom_files' in manifest.keys():
#         manifest['dicom_files'] = [ensure_unix_path_separators(fp) for fp in manifest['dicom_files']]
#     if 'nifti_file' in manifest.keys():
#         manifest['nifti_file'] = ensure_unix_path_separators(manifest['nifti_file'])
#     if 'rtss' in manifest.keys():
#         if manifest['rtss'] is not None:
#             manifest['rtss'] = ensure_unix_path_separators(manifest['rtss'])

#     blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#     container_name = 'input-dicoms'

#     file_paths = manifest['nifti_file']
#     if not verify_files(blob_service_client, container_name, file_paths):
#         return
    
#     if manifest['rtss'] is not None:
#         logging.info(f'rtss included in manifest, {manifest["rtss"]}')
#         if not verify_files(blob_service_client, container_name, manifest['rtss']):
#             return

#     # Prepare the message for the queue
#     queue_message = json.dumps(manifest)

#     msg.set(queue_message)

############################################################
    
@app.queue_trigger(arg_name="msg", queue_name="process-queue", connection="AzureWebJobsStorage")
def queue_trigger(msg: func.QueueMessage):
    # Deserialize the message
    manifest = json.loads(msg.get_body().decode('utf-8'))
    container_name_input = 'input-dicoms'
    container_name_output = 'output'

    connection_string = manifest['connection_string']
    account_name = manifest['account_name']
    account_key = manifest['account_key']

    # create folder structure for temp files
    base_temp_dir = tempfile.mkdtemp()
    # dicom_folder = os.path.join(base_temp_dir,manifest['dicom_folder'])

    if 'dicom_files' in manifest.keys():
        for dicom_file in tqdm(manifest['dicom_files'], desc=f'saving dicom file blobs as tmp files'): # these have paths relative input-dicoms container
            temp_file_path = save_blob_as_tmp(container_name_input, dicom_file, connection_string, base_temp_dir)

        # re-route path to dicom in manifest

        manifest['dicom_folder'] = os.path.join(base_temp_dir, manifest['dicom_folder'])

    if 'nifti_file' in manifest.keys():
        temp_file_path = save_blob_as_tmp(container_name_input, manifest['nifti_file'], connection_string, base_temp_dir)
        # save old nifti file path
        manifest['nifti_file_blob'] = manifest['nifti_file']
        # re-route paths to nifti in manifest for purpose of inference
        manifest['nifti_file'] = os.path.join(base_temp_dir, manifest['nifti_file'])

    if manifest['rtss'] is not None:
        temp_file_path_rtss = save_blob_as_tmp(container_name_input, manifest['rtss'], connection_string, base_temp_dir)
        # re-route paths to rtss in manifest
        manifest['rtss_tmp'] = os.path.join(base_temp_dir, manifest['rtss'])

    logging.info(f'tmp file paths = {glob.glob(f"{base_temp_dir}/**", recursive=True)}')
    
    manifest['base_temp_dir'] = base_temp_dir

    dl, inference_settings = setup_inference(manifest)

    DICOM_RTSS_OUTPUT = inference_settings['DICOM_RTSS_OUTPUT']
    NIFTI_RTSS = inference_settings['NIFTI_RTSS']
    NIFTI_IMAGE = inference_settings['NIFTI_IMAGE']
    NIFTI_GT = inference_settings['NIFTI_GT']
    DICE_PATH = inference_settings['DICE_PATH']
    CALIBRATION_PATH = inference_settings['CALIBRATION_PATH']

    do_inference(dl, inference_settings)

    logging.info(f'successfully performed inference')

    
    blob_name = f'{manifest["patient_ID"]}.zip'
    zip_buffer = create_zip_archive([f for f in [DICOM_RTSS_OUTPUT, NIFTI_IMAGE, NIFTI_RTSS, NIFTI_GT, DICE_PATH, CALIBRATION_PATH] if os.path.exists(f)])
    upload_zip_to_blob(zip_buffer, container_name_output, blob_name, connection_string)
    download_url = generate_sas_token(container_name_output, blob_name, account_name, account_key)

    # send_completion_email(container_name_output, zip_buffer, download_url, 'bjorn.victor.ahlgren@gmail.com', expiry_hours=72)
    
    if 'clean_up_files' in manifest.keys():
        file_paths = [f'manifest_{manifest["patient_ID"]}.yml']
        if manifest['clean_up_files']:
            if 'dicom_files' in manifest.keys():
                file_paths.append(manifest['dicom_files'])
            if 'nifti_file_blob' in manifest.keys():
                file_paths.append(manifest['nifti_file_blob'])
            if 'rtss' in manifest.keys():
                if manifest['rtss'] is not None:
                    file_paths.append(manifest['rtss'])
            delete_uploaded_files(container_name_input, file_paths, connection_string)

    logging.info(f'Done with patient {manifest["patient_ID"]}. download_url = {download_url}')
