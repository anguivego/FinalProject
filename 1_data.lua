----------------------------------------------------------------------
-- This script demonstrates how to load the (SVHN) House Numbers 
-- training data, and pre-process it to facilitate learning.
--
-- The SVHN is a typical example of supervised training dataset.
-- The problem to solve is a 10-class classification problem, similar
-- to the quite known MNIST challenge.
--
-- It's a good idea to run this script with the interactive mode:
-- $ th -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'dataset-mnist' -- function to load mnist dataset
----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end


-- geometry: width and height of input images
geometry = {32,32}

----------------------------------------------------------------------
----------------------------------------------------------------------
-- First we load the training data.

print '==> loading dataset'
-- geometry: width and height of input images
geometry = {32,32}

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}



--trainData = {
--   data = loaded.X:transpose(3,4),
--   labels = loaded.y[1],
--   size = function() return trsize end
--}
--
---- Finally we load the test data.
--
--
--loaded = torch.load(test_file,'ascii')
--testData = {
--   data = loaded.X:transpose(3,4),
--   labels = loaded.y[1],
--   size = function() return tesize end
--}
--

-- Visualization is quite easy, using itorch.image().

--if opt.visualize then
--   if itorch then
--   first256Samples_y = trainData.data[{ {1,256},1 }]
--   first256Samples_u = trainData.data[{ {1,256},2 }]
--   first256Samples_v = trainData.data[{ {1,256},3 }]
--   itorch.image(first256Samples_y)
--   itorch.image(first256Samples_u)
--   itorch.image(first256Samples_v)
--   else
--      print("For visualization, run this script in an itorch notebook")
--   end
--end
--