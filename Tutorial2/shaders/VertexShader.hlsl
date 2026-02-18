cbuffer Mat : register(b0)
{
    matrix ModelMatrix;
    matrix ModelViewMatrix;
    matrix InverseTransposeModelViewMatrix;
    matrix ModelViewProjectionMatrix;
};                       

struct VertexPosNormColor
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float3 Color : COLOR;
};

struct VertexShaderOutput
{
    float4 PositionVS : POSITION;
    float4 Color : COLOR;
    float4 Position : SV_Position;
};

VertexShaderOutput main(VertexPosNormColor IN)
{
    VertexShaderOutput OUT;

    OUT.Position = mul(ModelViewProjectionMatrix, float4(IN.Position, 1.0f));
    OUT.PositionVS = mul(ModelViewMatrix, float4(IN.Position, 1.0f));
    OUT.Color = float4(IN.Color, 1.0f);

    return OUT;
}