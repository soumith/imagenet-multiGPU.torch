--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then
      local options = opt -- make an upvalue to serialize over to donkey threads
      donkeys = Threads(
         opt.nDonkeys,
         function()
            require 'torch'
         end,
         function(idx)
            opt = options -- pass to all donkeys via upvalue
            tid = idx
            local seed = opt.manualSeed + idx
            torch.manualSeed(seed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            paths.dofile('donkey.lua')
         end
      );
   else -- single threaded data loading. useful for debugging
      paths.dofile('donkey.lua')
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end

if opt.classify ~= '' then --do classification
   local indRetrainDir = opt.retrain:find("/[^/]*$")
   local retrainDir = opt.retrain:sub(0, indRetrainDir)
   
   classes = torch.load(paths.concat(retrainDir, 'classes.t7'), classes)
   nClasses = #classes
   assert(nClasses, "Failed to get nClasses")
   print('nClasses: ', nClasses)
   
   nClassify = 0
   donkeys:addjob(function() return classifyLoader:size() end, function(c) nClassify = c end)
   donkeys:synchronize()
   assert(nClassify > 0, "Failed to get nClassify")
   print('nClassify: ', nClassify)
else
   nClasses = nil
   classes = nil
   donkeys:addjob(function() return trainLoader.classes end, function(c) classes = c end)
   donkeys:synchronize()
   nClasses = #classes
   assert(nClasses, "Failed to get nClasses")
   print('nClasses: ', nClasses)
   torch.save(paths.concat(opt.save, 'classes.t7'), classes)

   nTest = 0
   donkeys:addjob(function() return testLoader:size() end, function(c) nTest = c end)
   donkeys:synchronize()
   assert(nTest > 0, "Failed to get nTest")
   print('nTest: ', nTest)
end
