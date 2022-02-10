import logging
from data.data_modules import BiobankData
from data.data_modules_bellot import BiobankData_k
from data.data_modules_50k import BiobankData50k
from data.data_modules_big import BiobankData_big
from src.utils import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
  '-sc', '--scenario',
  type=str,
  help='Which dataset to use; bellot = Bellot preprocessing, 50k = top 50k snps, all = all snps from preprocessing.',
  required=True,
)
parser.add_argument(
  '-k', '--num_snps',
  type=int,
  default=10,
  help='For Bellot preprocessing. How many SNPs to use; 10 or 50',
  required=False,
)
args = parser.parse_args()

logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s %(message)s'
  )

logging.info(f'Scenario: {args.scenario}')
logging.info('Start')

if args.scenario == 'all':
    biobank_data = BiobankData(args=None,data_path=data_path,hdf5_path=hdf5_path,csv_path=csv_path)
elif args.scenario == '50k':
    biobank_data = BiobankData50k(args=None,data_path=data_path,hdf5_path=hdf5_path_50k,csv_path=csv_path)
elif args.scenario == 'bellot':
    biobank_data = BiobankData_k(args=None,num_k=args.num_snps,data_path=data_path,hdf5_path=hdf5_path_50k,csv_path=csv_path)
elif args.scenario == 'big':
    biobank_data = BiobankData_big(data_path=data_path,hdf5_path=hdf5_path_big)
else:
    logging.info('Not implemented.')

biobank_data.prepare_data()

logging.info('Set up the data')

#### For use with arguments


#
# parser.add_argument(
#     '-sex', '--sex',
#     type=str,
#     help='Sex for which to run the experiment (male or female)',
#     required=True,
# )
#
# logging.info(f'Sex: {args.sex}')
