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
--require('mobdebug').start()
require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

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
print '==> downloading dataset'

-- Here we download dataset files. 

-- Note: files were converted from their original Matlab format
-- to Torch's internal format using the mattorch package. The
-- mattorch package allows 1-to-1 conversion between Torch and Matlab
-- files.

-- The SVHN dataset contains 3 files:
--    + train: training data
--    + test:  test data
--    + extra: extra training data

-- By default, we don't use the extra training data, as it is much 
-- more time consuming



mnist = {}

mnist.path_remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
mnist.path_dataset = 'mnist.t7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train_32x32.t7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test_32x32.t7')

 if not paths.filep(mnist.path_trainset) or not paths.filep(mnist.path_testset) then
    local remote = mnist.path_remote
    local tar = paths.basename(remote)
    os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
 end
train_file = mnist.path_trainset
test_file = mnist.path_testset

if not paths.filep(train_file) then
   os.execute('wget ' .. www .. train_file)
end
if not paths.filep(test_file) then
   os.execute('wget ' .. www .. test_file)
end

----------------------------------------------------------------------
-- training/test size

if opt.size == 'full' then
   print '==> using regular, full training data'
   trsize = 60000
   tesize = 10000
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   trsize = 10000
   tesize = 2000
end

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, and re-arrange it to be compatible
-- with Torch's representation. Matlab uses a column-major representation,
-- Torch is row-major, so we just have to transpose the data.

-- Note: the data, in X, is 4-d: the 1st dim indexes the samples, the 2nd
-- dim indexes the color channels (RGB), and the last two dims index the
-- height and width of the samples.

loaded = torch.load(train_file,'ascii')
trainData = {
   data = loaded.data:transpose(3,4),
   labels = loaded.labels,
   size = function() return trsize end
}

-- If extra data is used, we load the extra file, and then
-- concatenate the two training sets.

-- Torch's slicing syntax can be a little bit frightening. I've
-- provided a little tutorial on this, in this same directory:
-- A_slicing.lua


-- Finally we load the test data.

loaded = torch.load(test_file,'ascii')
testData = {
   data = loaded.data:transpose(3,4),
   labels = loaded.labels,
   size = function() return tesize end
}

----------------------------------------------------------------------
print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

trainData.data = trainData.data:float()
testData.data = testData.data:float()


