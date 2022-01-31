import ctypes

from dogqc.cudalang import *
from dogqc.code import Code
import dogqc.identifier as ident
from enum import IntEnum
import copy


class KernelCall ( object ):

    defaultGridSize = 1024
    defaultBlockSize = 128

    def __init__ ( self, gridSize, blockSize ):
        self.blockSize = KernelCall.defaultBlockSize
        self.gridSize = KernelCall.defaultGridSize
        if blockSize != None:
            self.blockSize = blockSize
        if gridSize != None:
            self.gridSize = gridSize

    def generated ( kernel, gridSize=None, blockSize=None ):
        call = KernelCall ( gridSize, blockSize )
        call.kernel = kernel
        call.kernelName = kernel.kernelName
        return call 
    
    def library ( kernelName, parameters, templateParameters="", gridSize=None, blockSize=None ):
        call = KernelCall ( gridSize, blockSize )
        call.kernel = None
        call.kernelName = kernelName
        call.parameters = parameters
        call.templateParameters = templateParameters
        return call 

    def get ( self ):
        if self.kernel != None:
            return KernelCall.generic ( self.kernel.kernelName, self.kernel.getParameters(), self.gridSize, self.blockSize, kernel = self.kernel )
        else:
            return KernelCall.generic ( self.kernelName, self.parameters, self.gridSize, self.blockSize )

    def getAnnotations ( self ):
        if self.kernel != None and len(self.kernel.annotations) > 0:
            return " ".join(self.kernel.annotations)
        else:
            return ""


    def generic ( kernelName, parameters, gridSize=1024, blockSize=128, templateParams="", kernel = None ):
        # kernel invocation parameters
        code = Code()
    
        templatedKernel = kernelName
        if templateParams != "":
            templatedKernel += "<" + templateParams + ">"
    
        with Scope ( code ):
            emit ( "int gridsize=" + str(gridSize), code )
            emit ( "int blocksize=" + str(blockSize), code )
            kernel_call = templatedKernel + "<<<gridsize, blocksize>>>("

            # TODO(jigao): when calculate the size, re-check here to calculate the SHARED_MEMORY_USAGE
            if kernel != None and kernel.doGroup == True:
                num_bytes = ""
                for name, c in kernel.inputColumns.items():
                    # Only take the last aht and coming agg
                    if str.__contains__( name, "aht" ):
                        num_bytes = ""
                    if ( str.__contains__( name, "aht" ) or str.__contains__( name, "agg" ) ): # String match the hash aggregations.
                        dataType = c.dataType.replace( "agg_ht", "agg_ht_sm" )
                        if num_bytes == "":
                            num_bytes = sizeof( dataType )
                        else:
                            num_bytes = add( num_bytes, sizeof( dataType ) )
                emit ( assign( declareEasy( CType.INT, SHARED_MEMORY_USAGE ), mul( num_bytes, SHARED_MEMORY_HT_SIZE_CONSTEXPR_STR + kernel.deviceFunctionId )), code )
                emit ( cout() + " << \"Shared memory usage: \" << " + SHARED_MEMORY_USAGE + " << \" bytes\" << std::endl" , code )
                emit ( call(cudaFuncSetAttribute, [ kernelName, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEMORY_USAGE] ), code )
                kernel_call = templatedKernel + "<<<gridsize, blocksize," + SHARED_MEMORY_USAGE + ">>>("

            # add parameters: input attributes, output attributes and additional variables (output number)
            comma = False
            for a in parameters:
                if not comma:
                    comma = True
                else:
                    kernel_call += ", "
                kernel_call += str(a)
            kernel_call += ")"
            emit ( kernel_call, code )
        return code

    





class Kernel ( object ):
    
    def __init__ ( self, name ):
        self.init = Code()
        self.body = Code()
        self.inputColumns = {}
        self.outputAttributes = []
        self.variables = []
        self.kernelName = name
        self.annotations = []
        self.doGroup = False # if doGroup== True, then generate shared memory stuff inside of kernel as well as <<<,,>>> function call
        self.doCg = True # if true, use cg instead of atomicAdd in all threads.
        self.initVar_Map = {}
        self.deviceFunctionId = None

    def add ( self, code ):
        self.body.add( code )
    
    def addVar ( self, c ):
        # resolve multiply added columns
        self.inputColumns [ c.get() ] = c

    def getParameters ( self ):
        params = []
        for name, c in self.inputColumns.items():
            params.append ( c.getGPU() )
        for a in self.outputAttributes:
            params.append ( ident.gpuResultColumn( a ) )
        for v in self.variables:
            params.append( v.getGPU() )
        return params

    def initSM(self):
        assert self.doGroup
        init_sm = Code()

        offset = SHARED_MEMORY
        for name, c in self.inputColumns.items():
            if self.doGroup and ( str.__contains__( name, "aht" )  or str.__contains__( name, "agg" ) ):
                if str.__contains__( name, "aht" ):
                    # Only take the last aht and coming agg
                    init_sm = Code()
                    offset = SHARED_MEMORY
                    emit(declareexternSharedArrayEasy(CType.CHAR, SHARED_MEMORY), init_sm)
                dataType = c.dataType.replace("agg_ht", "agg_ht_sm")
                emit ( declareEasy( ptr( dataType ), name), init_sm )
                emit ( assign( name, cast( ptr( dataType ), offset ) ), init_sm )
                offset = add ( offset, mul ( sizeof( dataType ) , SHARED_MEMORY_HT_SIZE_CONSTEXPR_STR + self.deviceFunctionId ) )
                if str.__contains__( name, "aht" ) :
                    emit ( initSMAggHT( name, self.deviceFunctionId ), init_sm )
                elif str.__contains__( name, "agg" ):
                    emit ( initSMAggArray( name, self.deviceFunctionId, self.initVar_Map[name] ), init_sm )

        emit ( declareVolatileSharedEasy( CType.INT, HT_FULL_FLAG), init_sm )
        emit ( assign( HT_FULL_FLAG, intConst(0) ), init_sm )

        emit( syncthreads(), init_sm )
        self.init.addFront(init_sm)

    def getKernelCode( self ):
        kernel = Code()
        
        # open kernel frame
        kernel.add("__global__ void " + self.kernelName + "(")
        comma = False
        params = ""
        for name, c in self.inputColumns.items():
            if not comma:
                comma = True
            else:
                params += ", "
            assert c.get() == name
            if self.doGroup and ( str.__contains__( name, "aht" )  or str.__contains__( name, "agg" ) ):
                if str.__contains__( name, "aht" ) :
                    params = params.replace("g_aht", "aht")
                    params = params.replace("g_agg", "agg")
                params += c.dataType + "* g_" + c.get() # Use prefix "g_" as global ht
            else:
                params += c.dataType + "* " + c.get()
        for a in self.outputAttributes:
            params += ", " 
            params += a.dataType + "* " + ident.resultColumn( a )
        for v in self.variables:
            params += ", " 
            params += v.dataType + "* " + v.get()
        kernel.add( params + ") {")

        if self.doGroup:
            self.initSM()

        # add code generated by operator tree
        kernel.add(self.init.content)
        
        # add code generated by operator tree
        kernel.add(self.body.content)
        
        # close kernel frame
        kernel.add("}") 
        return kernel.content

    def annotate ( self, msg ):
        self.annotations.append(msg)


# For the only one device function: __device__ void sm_to_gm
class DeviceFunctionStatus ( IntEnum ):
    INIT    = 1 # not started
    STARTED = 2
    END  = 3

class DeviceFunction(object):
    def __init__(self, name):
        self.status = DeviceFunctionStatus.INIT
        self.init = Code()
        self.body = Code()
        self.inputColumns = {}
        # self.outputAttributes = []
        # self.variables = []
        self.functionName = name
        self.id = self.functionName[-1]
        # self.annotations = []

    def add(self, code):
        self.body.add(code)

    def addFront(self, code):
        self.body.addFront(code)

    def addVar(self, c):
        # resolve multiply added columns
        self.inputColumns[c.get()] = c

    # def getParameters(self):
    #     params = []
    #     for name, c in self.inputColumns.items():
    #         params.append(c.getGPU())
    #     for a in self.outputAttributes:
    #         params.append(ident.gpuResultColumn(a))
    #     for v in self.variables:
    #         params.append(v.getGPU())
    #     return params

    def getDeviceFunction(self):
        kernel = Code()

        # open kernel frame
        kernel.add("__device__ void " + self.functionName + "(")
        comma = False
        params = ""
        for name, c in self.inputColumns.items():
            if not comma:
                comma = True
            else:
                params += ", "
            params += c.dataType + "* " + c.get()
        # for a in self.outputAttributes:
        #     params += ", "
        #     params += a.dataType + "* " + ident.resultColumn(a)
        # for v in self.variables:
        #     params += ", "
        #     params += v.dataType + "* " + v.get()
        kernel.add(params + ") {")

        # add code generated by operator tree
        kernel.add(self.init.content)

        # add code generated by operator tree
        kernel.add(self.body.content)

        # close kernel frame
        kernel.add("}")
        return kernel.content

    # def annotate(self, msg):
    #     self.annotations.append(msg)

