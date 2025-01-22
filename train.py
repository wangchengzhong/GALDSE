import argparse
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from galdse.backbones.shared import BackboneRegistry, FrontendRegistry
from galdse.stochastic_process import StoProcessRegistry
from galdse.data_module import SpecsDataModule

from galdse.model import GaldSE
import torch
def get_argparse_groups(parser):
     groups = {} 
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="TinyNCSNpp")
          parser_.add_argument("--frontend", type=str, choices=FrontendRegistry.get_all_names(), default="MagMask")
          parser_.add_argument("--sto_process", type=str, choices=StoProcessRegistry.get_all_names(), default="RESSHIFT")
          parser_.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
          parser_.add_argument("--check_interval", type=int, default=6, help="validation interval")


          
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for GaldSE, pl.Trainer, the resshift module, the frontend DNN class, and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     frontend_cls = FrontendRegistry.get_by_name(temp_args.frontend)
     sto_process_cls = StoProcessRegistry.get_by_name(temp_args.sto_process)
     parser = pl.Trainer.add_argparse_args(parser)
     GaldSE.add_argparse_args(
          parser.add_argument_group("GaldSE", description=GaldSE.__name__))
     
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     frontend_cls.add_argparse_args(
          parser.add_argument_group("Frontend", description=frontend_cls.__name__)
     )
     sto_process_cls.add_argparse_args(
          parser.add_argument_group("StoProcess", description=sto_process_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)

     # Initialize logger, trainer, model, datamodule

     model = GaldSE(
          backbone=args.backbone, frontend=args.frontend, sto_process=args.sto_process, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['GaldSE']),
               **vars(arg_groups['StoProcess']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['Frontend']),
               **vars(arg_groups['DataModule'])
          }
     )

     print(f"\nkappa_{model.diffusion.kappa}_factor_{model.data_module.spec_factor}_exp_{model.data_module.spec_abs_exponent}_power_{model.diffusion.power}_minNoiseLevel_{model.diffusion.min_noise_level}_augment_{model.data_module.augment_type}\n")
     # Set up logger configuration
     logger = TensorBoardLogger(save_dir="logs", name="tensorboard")


     # Set up callbacks for logger
     callbacks = [ModelCheckpoint(dirpath=f"logs/{logger.version}", save_last=True, filename='{epoch}-last')]

     if args.num_eval_files:
          checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"logs/{logger.version}", 
               save_top_k=2, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
          checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"logs/{logger.version}", 
               save_top_k=2, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
          callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr]


     trainer = pl.Trainer.from_argparse_args(
          arg_groups['pl.Trainer'],
          strategy=DDPStrategy(find_unused_parameters=False), logger=logger,
          log_every_n_steps=30, num_sanity_val_steps=0,
          callbacks=callbacks,accelerator='gpu',# devices=[0, 1],
          resume_from_checkpoint=args.resume_from_checkpoint,
          check_val_every_n_epoch=args.check_interval,
          max_epochs=15000,
          # val_check_interval=3000
     )

     # Train model
     trainer.fit(model)