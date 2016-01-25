--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local testDataIterator = function()
   testLoader:reset()
   return function() return testLoader:get_batch(false) end
end

local batchNumber
local top1_center, loss
local testConf = opt.conf and optim.ConfusionMatrix(classes) or nil
local testBatchSize = math.floor(opt.batchSize / opt.batchChunks)
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0
   loss = 0
   if testConf then testConf:zero() end
   for i=1,nTest/testBatchSize do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * testBatchSize + 1
      local indexEnd = (indexStart + testBatchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:get(indexStart, indexEnd)
            return inputs, labels
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / nTest
   loss = loss / (nTest/testBatchSize) -- because loss is calculated per batch
   testLogger:add{
      ['% top1 accuracy (test set) (center crop)'] = top1_center,
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, loss, top1_center))

   if opt.conf then
      io.write('==> Saving training confusion matrix...'); io.flush()
      if opt.verboseConf then
         local confFile = io.open(paths.concat(opt.save, 'confusion.txt'), 'w')
         confFile:write('-- Training --------------------------------------------------------------------\n')
         confFile:write(trainConf:__tostring__())
         confFile:write('\n\n')
         confFile:write('\n-- Testing ---------------------------------------------------------------------\n')
         confFile:write(testConf:__tostring__())
         confFile:write('\n\n')
         confFile:close()
      end
      torch.save(paths.concat(opt.save, 'confusion.t7'), {trainConf, testConf})
      print(' Done!')
   end

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()
local print_every = 16 * opt.batchChunks

function testBatch(inputsCPU, labelsCPU)
   batchNumber = batchNumber + testBatchSize

   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, labels)
   cutorch.synchronize()
   local pred = outputs:float()

   loss = loss + err

   local _, pred_sorted = pred:sort(2, true)
   for i=1,pred:size(1) do
      local g = labelsCPU[i]
      if pred_sorted[i][1] == g then top1_center = top1_center + 1 end
   end
   if batchNumber % (testBatchSize * print_every) == 0 then
      print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
   end

   if testConf then testConf:batchAdd(outputs:float(), labelsCPU) end
end
