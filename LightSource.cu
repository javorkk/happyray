#include "CUDAStdAfx.h"
#include "RT/Primitive/LightSource.hpp"

GLOBAL void normalizeIntensities(AreaLightSourceCollection aLights)
{
    
    for(uint lightID = globalThreadId1D(); lightID < aLights.size(); lightID+= numThreads())
    {
        float probabilityRCP = aLights.getTotalWeight() / aLights.getWeight(lightID);
        AreaLightSource ls = aLights.getLightWithID(lightID);
        ls.intensity *= probabilityRCP;
    }
}


HOST void AreaLightSourceCollection::normalizeALSIntensities()
{
    if(mSize <= 1u)
        return;

    dim3 block ( 128 );
    dim3 grid  ( 180 );
    normalizeIntensities<<< grid, block>>>(*this);
}