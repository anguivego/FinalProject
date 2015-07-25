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
   trsize = 60000
   tesize = 10000
else
   trsize = 2000
   tesize = 1000
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
trainData = mnist.loadTrainSet(trsize, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(tesize, geometry)
testData:normalizeGlobal(mean, std)
