baseDir = 'C:\Users\ncwang\Documents\Rao_lab\TCIA_Scorecard\data\scans';
dirList = rdirlist(baseDir);

%for i = 1:length(dirList)
hasDCMs = false(size(dirList));
dirInfoCell = cell(size(dirList));
for i = 1:length(dirList)
    cDir = dirList{i};
    fList = dir(fullfile(cDir,'*.dcm'));
    if ~isempty(fList)
        hasDCMs(i) = true;
        dirInfoCell{i} = fList;
    end
       
end
dirInfoCell(~hasDCMs) = [];
%%
dicomInfoCell = cell(size(dirInfoCell));
fNames = {};
for i = 1:length(dirInfoCell)
    cDcmList = dirInfoCell{i};
    cFile = cDcmList(1);
    cDir = cFile.folder;
    cDcm = cFile.name;
    dcmInfo = dicominfo(fullfile(cDir,cDcm),'UseDictionaryVR',true);
    dcmInfo.folder = cDir;
    [cPath,cName,~] = fileparts(cDir);
    [cRPath,~,~] = fileparts(cPath);
    [~,TCGA_ID,~] = fileparts(cRPath);
    dcmInfo.name = cName;
    dcmInfo.TCGA_ID = TCGA_ID;
    dicomInfoCell{i} = dcmInfo;
    cFNames = fieldnames(dcmInfo);
    fNames = union(fNames,cFNames);
end
%%
% remove private tags
filtMask = cellfun(@isempty,regexpi(fNames,'^Private'));
filtFNames = fNames(filtMask);
modalTypes = {'t1_type','t2_type','flair_type','dwi_type'};
dimType = {'axi_type','cor_type','sag_type'};
contrastType = {'contrast_type'};
structFNames = [filtFNames',modalTypes,dimType,contrastType,{'modal_type'}];
structConst = vertcat(structFNames,cell(size(structFNames)));
dicomInfoStruct = repmat(struct(structConst{:}),size(dicomInfoCell));
badSeries = false(size(dicomInfoCell));
reqFields = {'SeriesDescription','EchoTime','RepetitionTime'}; % require a description for the series
for i = 1:length(dicomInfoCell)
    currDcm = dicomInfoCell{i};
    
    for fldN = 1:length(reqFields) % test to make sure we have all the required fields
        currRF = reqFields{fldN};
        if ~isfield(currDcm,currRF) || isempty(currDcm.(currRF))
            badSeries(i) = true;
        end
    end
    if badSeries(i) == false
        for j = 1:length(filtFNames)
            cFld = filtFNames{j};
            if isfield(currDcm,cFld)
                cVal = currDcm.(cFld);
                % fix magnetic field strength
                if strcmpi(cFld,'MagneticFieldStrength')
                    if cVal > 1000
                        cVal = cVal/10000;
%                         display(currDcm.(cFld))
                    end
                end
                
                
                dicomInfoStruct(i).(cFld) = cVal;
            end
        end
    end
end
dicomInfoStruct(badSeries) = [];
%% determine what modality each sequence is
tcgaIDs = {dicomInfoStruct.TCGA_ID};
[uIDs,~,uMap] = unique(tcgaIDs);
% cIDN = 2;
rxModSet = struct('t1','t1','t2','t2(?!*)','flair','flair','dwi','(diff|dw)');
%     rxAxSet = struct('axi','ax','cor','cor','sag','sag');
% dicom orientation tag
dcAxSet = struct('axi',[1;0;0;0;1;0],...
    'cor',[1;0;0;0;0;-1],...
    'sag',[0;1;0;0;0;-1]);

modMap = struct('t1',struct('t1_type',true,'contrast_type',false,'flair_type',false),...
    't1post',struct('t1_type',true,'contrast_type',true),...
    't2',struct('t2_type',true,'dwi_type',false,'flair_type',false),...
    'flair',struct('flair_type',true,'t1_type',false),...
    'dwi',struct('dwi_type',true));
modList = fieldnames(modMap);
ctMat = zeros(length(uIDs),length(modList));
TECell = cell(length(uIDs),length(modList));
TRCell = cell(length(uIDs),length(modList));
DescCell = cell(length(uIDs),length(modList));

for cIDN = 1:length(uIDs)
    
    % group the matches by patient scan set
    cSeqMask = ismember(uMap,cIDN);
    cSeqIdxs = find(cSeqMask);
    cSeqs = dicomInfoStruct(cSeqMask);
    
    cDescs = {cSeqs.SeriesDescription}';
%     cSeqs(emptyDescs) = [];
%     cDescs(emptyDescs) = []; 
%     cSeqIdxs(emptyDescs) = [];
    

    cNames = {cSeqs.name}';
%     [cDescs{emptyDescs}] = deal(''); % fill with empty strings instead of doubles
    
    rxMods = fieldnames(rxModSet);
    % figure out which modality is in the series description
    for j = 1:length(rxMods)
        cMod = rxMods{j};
        cRX = rxModSet.(cMod);
        matches = ~cellfun(@isempty,regexpi(cDescs,rxModSet.(cMod)));
        cFName = [cMod '_type'];
        for cIdx = 1:length(cSeqs)
            cSeqs(cIdx).(cFName) = matches(cIdx);
        end
    end
    % check the 'ContrastBolusAgent' tag for contrast status
    % initially going for empty vs not.  Don't know if that's sufficient
    % insufficient, going to exclude, no/none
    CBAMask = ~cellfun(@isempty,{cSeqs.ContrastBolusAgent});
    % add a search on the description for post or gd or gad
    conMatches = ~cellfun(@isempty,regexpi(cDescs,'(post| gd| gad)'));
    conMatches = conMatches & cellfun(@isempty,regexpi(cDescs,'(pre)'));
    contrastMask = CBAMask | reshape(conMatches,size(CBAMask));
    
    % make sure we're not including no contrast agent
    CBAStrs = {cSeqs(CBAMask).ContrastBolusAgent};
    cCheck = ~cellfun(@isempty,regexpi(CBAStrs,'^no'));
    contrastMask(cCheck) = false;
    for cIdx = 1:length(cSeqs)
        cSeqs(cIdx).contrast_type = contrastMask(cIdx);
    end
    
%     rxAxSet = struct('axi','ax','cor','cor','sag','sag');
    dcAxs = fieldnames(dcAxSet);
    % figure out which direction the scan is in the series description
    for j = 1:length(dcAxs)
        cAx = dcAxs{j};
        cDcT = dcAxSet.(cAx);
        cFName = [cAx '_type'];
        for cIdx = 1:length(cSeqs)
            cIOP = round(cSeqs(cIdx).ImageOrientationPatient);
            if ~isempty(cIOP) && (numel(cIOP) == numel(dcAxSet.(cAx)))...
                % check to see if it has the same IOP as we expect
                cMatch = all(dcAxSet.(cAx) == cIOP);
                cSeqs(cIdx).(cFName) = cMatch;
            else
                cSeqs(cIdx).(cFName) = false;
            end
        end
    end
    
%     modList = fieldnames(modMap);
    for j = 1:length(modList)
        cMod = modList{j};
        cFieldCheck = modMap.(cMod);
        cFieldList = fieldnames(cFieldCheck);
        cStatus = true(1,length(cSeqs)); % defaults to match, might want to change
        
        % check to see if it matches all the criteria
        % will prefer axial, then coronal, then sagittal
        for dcAxNo = 1:3
            cAx = dcAxs{dcAxNo};
            cDcT = dcAxSet.(cAx);
            cFName = [cAx '_type'];
            
            for cFN = 1:length(cFieldList)
                cField = cFieldList{cFN};
                matchSet = cellfun(@(a)a==cFieldCheck.(cField),{cSeqs.(cField)});
                cStatus = cStatus & matchSet;
            end
            
            if nnz(cStatus) == 0
                break
            end
            % check each sequence against the direction field
            matchSet = cellfun(@(a)a==true,{cSeqs.(cFName)});
            if nnz(matchSet&cStatus) > 0 % if at least one scan matches, take that one
                cStatus = matchSet&cStatus;
                break
            end
        end
        
        % add the modality to each of the ones that match
        matchIdxs = find(cStatus);
%         disp(cMod)
        ctMat(cIDN,j) = nnz(matchIdxs);
        TECell{cIDN,j} = [cSeqs(matchIdxs).EchoTime];
        TRCell{cIDN,j} = [cSeqs(matchIdxs).RepetitionTime];
        DescCell{cIDN,j} = {cSeqs(matchIdxs).SeriesDescription};
        for cMI = reshape(matchIdxs,1,[])
            cModType = cSeqs(cMI).modal_type;
            if isempty(cModType)
                cSeqs(cMI).modal_type = cMod;
%                 disp(cMod)
            elseif (ischar(cModType) && ~strcmp(cMod,cModType)) ...
                    || (iscell(cModType) && ~ismember(cMod,cModType))
                % start turning it into a cell array of strings
                % might mess things up...
                if ~iscell(cModType)
                    cSeqs(cMI).modal_type = {cModType, cMod};
                else                
                    cSeqs(cMI).modal_type = [cModType, {cMod}];
                end
                warning('Multiple modalities matched, %s',...
                    sprintf('%s ',cSeqs(cMI).modal_type{:})) 
            end
        end
    end
    
    % put the adjusted info back into the dicomInfoStruct
    dicomInfoStruct(cSeqIdxs) = cSeqs;
end
%%
t2FTab = struct2table(dicomInfoStruct(strcmp({dicomInfoStruct.modal_type},'t2')));
figure, axH = gca;
for mI = 1:length(modList)
    cMod = modList{mI};
    if strcmp(cMod,'dwi')
        continue
    end
    cFTab = struct2table(dicomInfoStruct(strcmp({dicomInfoStruct.modal_type},cMod)));
    TE = cFTab.EchoTime;
    TR = cFTab.RepetitionTime;
    histogram2(axH,TE,TR,20)
    if mI == 1
        hold on
        axis vis3d, xlabel('TE'),ylabel('TR')
    end
    figure,histogram2(TE,TR,20)
    title(cMod)
    xlabel('TE'),ylabel('TR')
end
legend(axH,modList(1:4))
%%
MeanTEVals = nan(size(TECell));
MinTEVals  = nan(size(TECell));
MaxTEVals  = nan(size(TECell));
TEMask = ~cellfun(@isempty,TECell);
MeanTEVals(TEMask) = cellfun(@mean,TECell(TEMask));
MinTEVals(TEMask) = cellfun(@min,TECell(TEMask));
MaxTEVals(TEMask) = cellfun(@max,TECell(TEMask));
xlswrite('TE_TR.xlsx',[{''} modList';uIDs' num2cell(MeanTEVals)],'MeanTE')
xlswrite('TE_TR.xlsx',[{''} modList';uIDs' num2cell(MinTEVals)],'MinTE')
xlswrite('TE_TR.xlsx',[{''} modList';uIDs' num2cell(MaxTEVals)],'MaxTE')


MeanTRVals = nan(size(TRCell));
MinTRVals  = nan(size(TRCell));
MaxTRVals  = nan(size(TRCell));
TRMask = ~cellfun(@isempty,TRCell);
MeanTRVals(TRMask) = cellfun(@mean,TRCell(TRMask));
MinTRVals(TRMask) = cellfun(@min,TRCell(TRMask));
MaxTRVals(TRMask) = cellfun(@max,TRCell(TRMask));
xlswrite('TE_TR.xlsx',[{''} modList';uIDs' num2cell(MeanTRVals)],'MeanTR')
xlswrite('TE_TR.xlsx',[{''} modList';uIDs' num2cell(MinTRVals)],'MinTR')
xlswrite('TE_TR.xlsx',[{''} modList';uIDs' num2cell(MaxTRVals)],'MaxTR')
%% Filter by scans that are in the 1p/19q project
boxDir = 'C:\Users\ncwang\Box\1p19q Project';
dataDir = fullfile(boxDir,'data','data_split_n4');
id_T = readtable(fullfile(dataDir,'label_idstr.csv'));
id_strs = unique(table2cell(id_T(:,1)));

tcgaIDs = {dicomInfoStruct.TCGA_ID}';
[uIDs,~,uMap] = unique(tcgaIDs);
missing_IDs = cellfun(@isempty,tcgaIDs); % check for missing IDs
nonIDs = ~ismember(id_strs,tcgaIDs);

matchMask = ismember(tcgaIDs,id_strs);
nonMatchIDs = unique(tcgaIDs(~matchMask));
nonMatchUMask = ismember(uIDs,nonMatchIDs);
matchUMask = ismember(uIDs,unique(tcgaIDs(matchMask)));
matchUIdxs = find(matchUMask);
fDicomInfoStruct = dicomInfoStruct(matchMask);
exNo = 11;
% [matchUIdxs', ctMat(matchUIdxs,:)]
% cSeqs = dicomInfoStruct(ismember(tcgaIDs,uIDs{matchUIdxs(exNo)}));
%% SMASH THE DATA TOGETHER WHO CARES
nifti_dir = 'C:\Users\ncwang\Box\1p19q Project\data\N4_corrected_image_data\';

modalsToFind = {'t1','t1post','t2','flair'};
% infoFields = {'TCGA_ID','SeriesDescription','modal_type','Filename','NumSeries'};
infoFields = {'TCGA_ID','SeriesDescription','modal_type','NumSeries'};
dcmFields = {'EchoTime','RepetitionTime','Manufacturer','ManufacturerModelName',...
    'MagneticFieldStrength', };
otherFields = {'TumorVolume'};
structFNames = [infoFields,dcmFields];
structConst = vertcat(structFNames,cell(size(structFNames)));
outStruct = repmat(struct(structConst{:}),length(id_strs),length(modalsToFind));

for mNo = 1:length(modalsToFind)
    cMod = modalsToFind{mNo};
    cModMatches = strcmp(cMod,{dicomInfoStruct.modal_type});
    for idNo = 1:length(id_strs)
        cIDStr = id_strs{idNo};
        outStruct(idNo,mNo).TCGA_ID = cIDStr;
        outStruct(idNo,mNo).modal_type = cMod;
        
        cIDMatches = strcmp(cIDStr,{dicomInfoStruct.TCGA_ID});
        cMask = cIDMatches & cModMatches;
        cIdxs = find(cMask);
        cSeqs = dicomInfoStruct(cMask);
        outStruct(idNo,mNo).NumSeries = nnz(cMask);
        if nnz(cMask) > 0
%             cFilenames = {cSeqs.Filename};
%             outStr = sprintf('%s|',cFilenames{:}); outStr(end) =[];
%             outStruct(idNo,mNo).Filename = outStr;
            cSerDescs = {cSeqs.SeriesDescription};
            outStr = sprintf('%s|',cSerDescs{:}); outStr(end) =[];
            outStruct(idNo,mNo).SeriesDescription = outStr;
            % loop through the fields we've requested
            for fNo = 1:length(dcmFields)
                cFN = dcmFields{fNo};
                cMat = {cSeqs.(cFN)};
                cMat(cellfun(@isempty,cMat)) = [];
                if isempty(cMat)
                    continue
                end
                if isnumeric(cMat{1}) % if numeric take the mean
                    cVal = mean(cell2mat(cMat));
                else % pick the most common entry
                    [uVal,~,ic] = unique(cMat);
                    currCt = accumarray(ic,1);
                    [~,order] = sort(currCt,'descend');
                    cVal = uVal{order(1)};
                end
                outStruct(idNo,mNo).(cFN) = cVal;
            end
        else
%             outStruct(idNo,mNo).Filename = '';
            outStruct(idNo,mNo).SeriesDescription = '';
        end
        

    end
end

for idNo = 1:length(id_strs)
    cIDStr = id_strs{idNo};
    % load up the nifti files, to see if I can pull out the tumor
    % volume
    niiFName = fullfile(nifti_dir,cIDStr,'truth.nii.gz');
    disp(cIDStr)
    if exist(niiFName,'file')
        labelNII = niftiread(niiFName);
        [outStruct(idNo,:).TumorVolume] = deal(nnz(labelNII));
    else
        [outStruct(idNo,:).TumorVolume] = deal(nan);
    end
end

for mNo = 1:length(modalsToFind)
    cMod = modalsToFind{mNo};
    cTab = struct2table(outStruct(:,mNo));
    writetable(cTab,fullfile(boxDir,'data',sprintf('%s_dcminfo.csv',cMod)))
end

%%
% cFldr = 'C:\Users\ncwang\Documents\Rao_lab\TCIA_Scorecard\data\scans\TCGA-LGG\TCGA-CS-4938\04-15-1996-JHN BRAIN MR OP-84760';
% 
% niiInfo = niftiinfo(fullfile(nifti_dir,cIDStr,'truth.nii.gz'));

% 
% fList = dir(fullfile(cFldr,'*.dcm'))
% dcmInfo = dicominfo(fullfile(cFldr,'000000.dcm'));