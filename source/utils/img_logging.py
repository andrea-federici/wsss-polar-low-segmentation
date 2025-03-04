import torch
from torchvision.utils import make_grid
# from contextlib import contextmanager

# from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
# from pytorch_lightning.loggers import NeptuneLogger


# @contextmanager
# def evaluating(net):
#     '''Temporarily switch to evaluation mode.'''
#     istrain = net.training
#     try:
#         net.eval()
#         yield net
#     finally:
#         if istrain:
#             net.train()


def xy_grid(x, y, y_pred):

    xy = torch.concat([x, y.unsqueeze(1)], axis=1)
    y_pred_thresh = torch.argmax(y_pred, dim=1, keepdim=True)
    xy = torch.concat([xy, y_pred_thresh], axis=1)

    grid = make_grid(xy, pad_value=.5, nrow=16)
    img = torch.concat([grid[i:i+1,:,:] for i in range(grid.shape[0])], axis=1)

    return img


# def images(writer, dataloader, title, model=None, device=None, epoch=None): 

#     if device is None: 
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     x, y = next(iter(dataloader))
#     x = x.to(device)
#     y = y.unsqueeze(1).to(device)

#     xy = torch.concat([x, y], axis=1)

#     if model is not None: 
#         with evaluating(model) as m: 
#             with torch.no_grad():
#                 pred = torch.sigmoid(m(x)) >= .5
#                 # pred = m(x)
#                 xy = torch.concat([xy, pred], axis=1)
        
#     grid = make_grid(xy, pad_value=.5, nrow=16)
#     img = torch.concat([grid[i:i+1,:,:] for i in range(grid.shape[0])], axis=1)
    
#     if isinstance(writer, TensorBoardLogger):
#         writer.add_image(title, img, epoch)
#     elif isinstance(writer, NeptuneLogger):
#         img = img.permute(1,2,0)
#         writer.log_tensor_img(img, title+"_ep"+str(epoch))
#     else:
#         raise TypeError("Logger type not OK:", type(writer))



# def rgbs(writer, dataloader, title, model=None, epoch=None, device=None, in_channels=None): 

#     if device is None: 
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     if in_channels is None: 
#         in_channels = slice(None)

#     x, y = next(iter(dataloader))

#     x = x.to(device)
#     y = y.unsqueeze(1).to(device)

#     model.eval()

#     with torch.no_grad():
#         pred = torch.sigmoid(model(x)) >= .5
        
#         # Inputs: 
#         grgb = make_grid(x[:,in_channels,:,:], pad_value=.5, nrow=x.shape[0])
        
        
#         # Predictions: 
#         if model is None: 
#             img = grgb
#         else: 
#             gpred = make_grid(torch.concat([y, pred, pred], axis=1), pad_value=.5, nrow=y.shape[0])
#             img = torch.concat([grgb, gpred], axis=1)

#         # Log image: 
#         if isinstance(writer, TensorBoardLogger):
#             writer.add_image(title, img, epoch)
#         elif isinstance(writer, NeptuneLogger):
#             writer.log_figure(img, title)

