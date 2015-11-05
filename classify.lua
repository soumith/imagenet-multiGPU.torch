--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
classifyLogger = optim.Logger(paths.concat(opt.save, 'classify.log'))

local testDataIterator = function()
   classifyLoader:reset()
   return function() return classifyLoader:get_batch(false) end
end

local batchNumber
local timer = torch.Timer()
local predFile = assert(io.open(paths.concat(opt.classify, 'predictions.txt'), 'w'))


function classify()
   print('==> classifying data:')
   
   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()


   -- make sure we enter the loop by adjusting batchSize
   opt.batchSize = math.min(opt.batchSize, nClassify)

   for i=1,nClassify/opt.batchSize do -- nClassify is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = classifyLoader:get(indexStart, indexEnd)
            return inputs
         end,
         -- callback that is run in the main thread once the work is done
         classifyBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()
   io.close(predFile)

end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()

function classifyBatch(inputsCPU)
   batchNumber = batchNumber + opt.batchSize

   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   
   local outputs = model:forward(inputs)
   cutorch.synchronize()
   local pred = outputs:float()
   
   local _, pred_sorted = pred:sort(2, true)
   for i=1,pred:size(1) do
      predFile:write(trainLoader.classes[pred_sorted[i][1]], '\n')
   end

   if batchNumber % 1024 == 0 then
      print(('Epoch: Classifying [%d][%d/%d]'):format(epoch, batchNumber, nClassify))
   end
end
