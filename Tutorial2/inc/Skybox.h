#pragma once

#include <DirectXMath.h>

using namespace DirectX;

class Skybox
{
public:
    struct SkyboxVertex
    {
        XMFLOAT3 Position;
    };

    static const SkyboxVertex g_SkyboxVertices[];
    static const uint16_t g_SkyboxIndices[];
};

Skybox::SkyboxVertex const Skybox::g_SkyboxVertices[] =
{
    { {-1, -1, -1} }, { {-1, +1, -1} }, { {+1, +1, -1} }, { {+1, -1, -1} }, // front
    { {-1, -1, +1} }, { {-1, +1, +1} }, { {+1, +1, +1} }, { {+1, -1, +1} }, // back
};

const uint16_t Skybox::g_SkyboxIndices[] =
{
    0,1,2, 0,2,3, // front
    4,6,5, 4,7,6, // back
    4,5,1, 4,1,0, // left
    3,2,6, 3,6,7, // right
    1,5,6, 1,6,2, // top
    4,0,3, 4,3,7, // bottom
};