require('pl')
require('xlua')
require('image')
local dir = require('pl.dir')
local path = require('pl.path')


local opt = lapp([[
Required parameters
   --src            (default '.')     path for source image root directory
   --dst            (default '.')     path for destination image root directory

(Optional) parameters
   --dim            (default 256)     output image size
   --ratio          (default 0.1)     training/validation set split ratio [0 0.5]
   --inner_crop     (default true)    region of crop (inner or outer square box)
   --offset         (default 0)       offset for image boundary when outer_crop used
   --training_dir   (default 'train') output directory name for training set
   --validation_dir (default 'val')   output directory name for validation set
]])


local function has_image_extensions(filename)
   -- grab file extension
   local ext = string.lower(path.extension(filename))

   -- compare with list of image extensions
   local img_extensions = {'.jpeg', '.jpg', '.png', '.ppm', '.pgm'}
   for i = 1, #img_extensions do
      if ext == img_extensions[i] then
         return true
      end
   end
   return false
end


local function resize_image(dst, src, dim, inner_crop, offset)
   local crop_mode = (inner_crop and ('^'..dim)) or tostring(dim)
   local outer_crop = not inner_crop


   -- load and rescale image from src
   local x = image.load(src)
   x = image.scale(x, crop_mode)

   -- consider 2-dim image
   x = ((x:dim() == 2) and x:view(1, x:size(1), x:size(2))) or x

   -- consider greyscale image
   x = ((x:size(1) == 1) and x:repeatTensor(3,1,1)) or x

   -- consider RGBA image
   x = ((x:size(1) > 3) and x[{{1,3},{},{}}]) or x


   -- calculate coordinate for crop (left top of box)
   local lbox = math.floor(math.abs(x:size(3) - dim)/2 + 1)
   local tbox = math.floor(math.abs(x:size(2) - dim)/2 + 1)

   -- copy paste to y depending on crop_mode
   local y
   if inner_crop then
      y = x[{{},{tbox,tbox+dim-1},{lbox,lbox+dim-1}}]
   elseif outer_crop then
      y = torch.Tensor():typeAs(x):resize(3, dim, dim):fill(offset)
      y[{{},{tbox,tbox+x:size(2)-1},{lbox,lbox+x:size(3)-1}}]:copy(x)
   end


   -- save image to dst path
   image.save(dst, y)
end


local function create_dataset(arg)
   local arg = arg or {}

   -- arguments for image src and dst (they must be different)
   local dst_root = arg.dst or '.'
   local src_root = arg.src or '.'
   assert(dst_root ~= src_root, 'No overwrite allowed (src/dst paths are the same')

   -- argument for image output dimensions
   local dim = arg.dim or 256

   -- arguments for train/val set split (name and ratio)
   local ratio = math.min(math.max(arg.ratio, 0), 0.5) or 0
   local training_dir   = arg.training_dir or 'train'
   local validation_dir = arg.validation_dir or 'val'

   -- arguments for crop style (inner_crop or outer_crop)
   local inner_crop = arg.inner_crop or true
   local offset = arg.offset or 0

   print('==> parameters for dataset creation')
   print(arg)


   -- create dst root directory
   dir.makepath(dst_root)

   -- count #directories to search
   local cmd = "find "..src_root.." -name '*.JPG' -o -name '*.jpg' -o -name '*.png' -o -name '*.PNG' -o -name '*.JPEG' -o -name '*.jpeg' | wc -l"
   local total_imgs = tonumber(io.popen(cmd):read())

   -- flatten all (sub) directories and traverse one-by-one
   print('==> processing in progress')
   local src_path, dst_path
   local nb_image_processed = {train = 0, val = 0}
   for loc, dirs, files in dir.walk(src_root, false, false) do

      -- (1) get src/dst path
      if src_path == nil then
         src_path = loc   -- remember base path (src root)
         dst_path = ''    -- consider exception at src_root
      else
         dst_path = string.sub(loc, string.len(src_path)+2)
      end


      -- (2) remove non-images from file list
      for i = #files, 1, -1 do
         if not has_image_extensions(files[i]) then
            table.remove(files, i)
         end
      end


      -- (3) create dst directories (train/val)
      if #files > 0 then            -- create train directory only if output exists
         dir.makepath(path.join(dst_root, training_dir, dst_path))
      end
      if (ratio > 0) and (#files > 0) then
         dir.makepath(path.join(dst_root, validation_dir, dst_path))
      end


      -- (4) resize img and split into train/val sets
      local shuffle = (#files > 0) and torch.randperm(#files)  -- idx for split
      local corrupted = {}
      for i = 1, #files do
         -- train/val directory selector based on shuffled index
         local mode = ((i <= ratio*#files) and validation_dir) or training_dir

         -- specify each src/dst of images and resize
         local src = path.join(loc, files[ shuffle[i] ])
         local dst = path.join(dst_root, mode, dst_path, path.basename(files[ shuffle[i] ]))
         s = pcall(function() resize_image(dst, src, dim, inner_crop, offset) end)
         if not s then table.insert(corrupted, src) end

         nb_image_processed[mode] = nb_image_processed[mode] + 1
         xlua.progress(nb_image_processed[training_dir]+nb_image_processed[validation_dir], total_imgs)
         if (i % 10 == 0) then
            collectgarbage()
         end
      end

      if #corrupted > 0 then
         print('There are ' .. #corrupted .. ' corrupted images\n')
         for _, c in ipairs(corrupted) do print(c) end
         print('')
      end

      -- (5) force to have at least one sample in validation set (produce duplicated sample)
      if (#files > 0) and (ratio*#files < 1) and (ratio > 0) then
         local mode = validation_dir
         local file_idx = math.random(#files)

         local src = path.join(loc, files[file_idx])
         local dst = path.join(dst_root, mode, dst_path, path.basename(files[file_idx]))
         resize_image(dst, src, dim, inner_crop, offset)
         if not s then print(src .. ' has not been added as extra image to ' .. dst) end
         nb_image_processed[mode] = nb_image_processed[mode] + 1
      end

      -- jump to next directory
   end
   return nb_image_processed
end


local cnt = create_dataset(opt)
print('==> processed images')
print(cnt)
