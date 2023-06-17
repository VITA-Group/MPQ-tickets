import os
import sys

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

torch.set_default_dtype(torch.float64)


class lth:
    def __init__(self, model, percentage, mask=None, module_list=None):
        self.model = model
        self.percentage = percentage
        self.module_list = module_list
        if mask is None:
            self.create_mask()
        else:
            self.mask = mask
        self.init_state_dict = None


    def create_mask(self):
        self.mask = self.init_mask(self.model)

    @classmethod
    def init_mask(cls, model):
        mask = {}
        for name, param in model.named_parameters():
            if 'weight' in name and 'conv' in name:
                mask[name] = torch.ones_like(param) * 32
        return mask

    def get_mask(self):
        assert self.mask, 'mask not exists'
        avg_bit = self.get_avg_bit_of_mask(self.mask)
        return self.mask, avg_bit

    @classmethod
    def get_avg_bit_of_mask(cls, mask):
        masks = []
        for param in mask.values():
            masks.append(param.reshape(-1).cpu().numpy())
        return round(np.mean(np.concatenate(masks)), 2)

    def generate_new_mask(self):
        '''
            ATTENTION
            This is not a pure function
            this function will update mask inside the model
            return mask, avg_bit
        '''
        new_mask = self.mask
        model_state_dict = self.model.state_dict()

        all_value = [param.data.cpu().numpy() for name, param in self.model.named_parameters() if name in self.mask.keys()]
        alive = np.concatenate([e.flatten() for e in [tensor[np.nonzero(tensor)] for tensor in all_value]])
        percentile_value = np.percentile(abs(alive), self.percentage)

        for name in new_mask.keys():
            mask_val = new_mask[name]


            param = model_state_dict[name]
            weight_dev = param.device

            tensor = param.data.cpu().numpy()
            # alive = tensor[np.nonzero(tensor)]
            # percentile_value = np.percentile(abs(alive), self.percentage)

            new_mask[name] = torch.where(
                torch.from_numpy(abs(tensor) < percentile_value).to(weight_dev),
                torch.div(mask_val, 2).trunc().to(weight_dev),
                mask_val
            )

            mask_tmp=(new_mask[name]==1).int()+(new_mask[name]==2).int()
            new_mask[name]=(1-mask_tmp)*new_mask[name]

        self.mask = new_mask

        for name,param in self.model.named_parameters():
            if name in self.mask.keys():
                param.data=(self.mask[name]!=0).int()*param.data

        return new_mask, self.get_avg_bit_of_mask(new_mask)

    def get_lth_stats(self):
        raise NotImplementedError

    @classmethod
    def generate_random_mask(cls):
        new_mask = self.mask

        for name, param in self.model.named_parameters():
            if name in self.mask.keys():
                param.data = torch.randn(param.data.shape).to(param.device)


        all_value = [param.data.cpu().numpy() for name, param in self.model.named_parameters() if name in self.mask.keys()]
        alive = np.concatenate([e.flatten() for e in [tensor[np.nonzero(tensor)] for tensor in all_value]])
        percentile_value = np.percentile(abs(alive), self.percentage)


        for name, param in self.model.named_parameters():
            if name in self.mask.keys():
                mask_val = new_mask[name]
                weight_dev = param.device

                tensor = param.data.cpu().numpy()
                # alive = tensor[np.nonzero(tensor)]
                # percentile_value = np.percentile(abs(alive), self.percentage)

                new_mask[name] = torch.where(
                    torch.from_numpy(abs(tensor) < percentile_value).to(weight_dev),
                    torch.div(mask_val, 2).trunc().to(weight_dev),
                    mask_val
                )

                mask_tmp = (new_mask[name] == 1).int() + (new_mask[name] == 2).int()
                new_mask[name] = (1 - mask_tmp) * new_mask[name]

        self.mask = new_mask

        for name,param in self.model.named_parameters():
            if name in self.mask.keys():
                param.data=(self.mask[name]!=0).int()*param.data

        return new_mask, self.get_avg_bit_of_mask(new_mask)


if __name__ == '__main__':
    import torchvision.models as m

    model = m.resnet18()
    lt = lth(model=model, percentage=10)

    mask = lt.generate_new_mask()

    _, avg_bit = lt.get_mask()
    expected_avg_bit = 0.9 * 32 + 0.1 * 16
    assert round(avg_bit, 1) == round(expected_avg_bit, 1), 'test failed!'

    print('Congs, we did it.')
