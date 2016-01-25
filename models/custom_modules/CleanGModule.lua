require 'nn'
require 'nngraph'

local CleanGModule, parent = torch.class('nn.CleanGModule','nn.gModule')

local function recursiveFree(input)
      if torch.isTensor(input) then
         input:set()
      elseif torch.type(input) == 'table' then
         for i=1,#input do
            recursiveFree(input[i])
         end
      else
         error('recursiveFree found an unsupported type: ' .. torch.type(input))
      end
end

function CleanGModule:backward(input, gradOutput, scale)
   scale = scale or 1
   self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput, scale)
   return self.gradInput
end

function CleanGModule:updateGradInput(input, gradOutput)
   nn.gModule.updateGradInput(self, input, gradOutput)
   -- :set() will prevent the whole gModule's gradInput from being freed
   self.gradInput = torch.Tensor():typeAs(self.gradInput):set(self.gradInput)
end

function CleanGModule:accGradParameters(input,gradOutput,lr)
   local function neteval(node)
      if node.data.module then
         local module = node.data.module
         local gradOutput = node.data.gradOutput[1]
         if #node.data.gradOutput > 1 then
            gradOutput = node.data.gradOutputBuffer
         end
         local input = node.data.input
         -- a parameter node is captured
         if input == nil and node.data.module ~= nil then
            input = {}
         end
         if #input == 1 then
            input = input[1]
         end
         -- accGradParameters through this node
         module:accGradParameters(input,gradOutput,lr)
         -- free the unnecessary memory (cudnn modules are kind of special)
         if not (string.find(torch.type(node.data.module), 'cudnn')) then
            recursiveFree(node.data.gradOutput)
            if node.data.gradOutputBuffer then
               recursiveFree(node.data.gradOutputBuffer)
            end
         end
      end
      if self.verbose then
         print(' V : ' .. node:label())
      end
   end
   local outnode = self.outnode
   if #outnode.children > 1 and #gradOutput ~= #outnode.children then
      error(string.format('Got %s gradOutputs instead of %s', #gradOutput, #outnode.children))
   end
   for i,node in ipairs(self.backwardnodes) do
      neteval(node)
   end

   local function cleanGradInput(node)
      if node.data.module then
         if not (string.find(torch.type(node.data.module), 'cudnn')) then
            recursiveFree(node.data.module.gradInput)
         end
      end
   end
   for i,node in ipairs(self.backwardnodes) do
      -- the last node should have it's gradInput untouched
      if i < #self.backwardnodes then
         cleanGradInput(node)
      end
   end
   collectgarbage()
end
