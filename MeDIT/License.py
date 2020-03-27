import os
import datetime
import truepy

from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization

from truepy import LicenseData, License
from MeDIT.Others import GetPhysicaladdress

class MyLicense:
    def __init__(self, ssl_path=r'C:\MyProgram\Git\usr\bin\openssl.exe'):
        self._ssl_path = ssl_path
        self._password = ''

    def GenerateKeyAndCertification(self, store_folder='', validate_days=0):
        if store_folder:
            certificate_path = os.path.join(store_folder, 'certificate.pem')
        else:
            certificate_path = 'certificate.pem'

        if store_folder:
            key_path = os.path.join(store_folder, 'key.pem')
        else:
            key_path = 'key.pem'

        cmd = '"{:s}" req -x509 -newkey rsa:4096 -keyout {:s} -out {:s} -days {:d}'.format(self._ssl_path, key_path, certificate_path, validate_days)
        os.system(cmd)

    def IssueLicense(self, key_folder, licence_input, password='ahmFVThGrEAZeF', licence_file='license.key', validate_days=365):
        with open(os.path.join(key_folder, 'certificate.pem'), 'rb') as f:
            certificate = f.read()

        # Load the private key
        with open(os.path.join(key_folder, 'key.pem'), 'rb') as f:
            key = serialization.load_pem_private_key(
                f.read(),
                password=str.encode(password, encoding='UTF-8'),
                backend=backends.default_backend())

        # Issue the license
        license = License.issue(
            certificate,
            key,
            license_data=LicenseData(
                datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                (datetime.datetime.now() + datetime.timedelta(days=validate_days)).strftime('%Y-%m-%dT%H:%M:%S')))

        # Store the license
        with open(licence_file, 'wb') as f:
            license.store(f, str.encode(licence_input, encoding='UTF-8'))

    def VerifyLicense(self, certificate_path, licence_input, licence_file='license.key'):
        with open(certificate_path, 'rb') as f:
            certificate = f.read()

        # Load the license
        try:
            with open(licence_file, 'rb') as f:
                license = License.load(f, str.encode(licence_input, encoding='UTF-8'))
                license.verify(certificate)
            return True
        except truepy.License.InvalidPasswordException:
            return False

if __name__ == '__main__':
    syl = MyLicense()
    # syl.GenerateKeyAndCertification(r'C:\MyCode\FAE\LicenceKey', 1000)
    syl.IssueLicense(r'C:\MyCode\FAE\LicenseKey', GetPhysicaladdress(), licence_file=r'C:\MyCode\FAE\LicenseKey\license.key')
    # if syl.VerifyLicense(r'C:\MyCode\FAE\LicenseKey', GetPhysicaladdress()):
    #     print(1)
    # else:
    #     print(2)