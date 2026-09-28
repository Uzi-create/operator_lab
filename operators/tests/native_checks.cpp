#include "../operators.cpp"
#include <cassert>
#include <random>

int main() {
    std::mt19937 rng(42);
    for(int w:{1,2,3,7,19})for(int h:{1,2,5,13}) {
        std::vector<float> input(w*h),output(w*h);
        for(auto& value:input)value=float(rng()%1001)/1000;
        for(int mode=0;mode<=9;++mode)for(int radius:{0,2,7,2147483647}) {
            if(mode==5 && radius<2)continue;
            assert(run_operator(input.data(),output.data(),w,h,radius,mode,mode==5?2.f:.12f)==0);
            for(float value:output)assert(std::isfinite(value)&&value>=0&&value<=1);
        }
        for(int mode=0;mode<=3;++mode) {
            assert(run_geometry(input.data(),output.data(),w,h,mode,mode==3?8.f:.08f,.2f)==0);
            for(float value:output)assert(!std::isnan(value));
        }
        assert(metal_ridge(input.data(),output.data(),w,h)==0);
        for(float value:output)assert(std::isfinite(value)&&value>=0&&value<=1);
        std::vector<double> values(input.begin(),input.end());
        std::vector<uint8_t> domain(w*h,1),valid(w*h);
        std::vector<int32_t> counts(w*h),capacities(w*h);
        std::vector<float> mean(w*h),variance(w*h),residual(w*h),score(w*h);
        void* workspace=stats_create();assert(workspace);
        for(int outer:{0,1,12,4096}) {
            assert(masked_stats(workspace,values.data(),domain.data(),w,h,-1,outer,.02,1,.5,
                mean.data(),variance.data(),residual.data(),score.data(),valid.data(),counts.data(),capacities.data())==0);
            for(size_t i=0;i<values.size();++i) {
                assert(valid[i] && counts[i]==capacities[i]);
                assert(std::isfinite(score[i]) && variance[i]>=0);
            }
        }
        stats_destroy(workspace);
        std::vector<unsigned char> mask(w*h,1),support(w*h);
        double config[12]={.08,1.5,.03,3,.3,.65,.008,.8,200,200,0,0},result[14];
        int status=estimate_grasp_depth(input.data(),mask.data(),w,h,config,result,support.data());
        assert(status==0 || status==3 || status==4 || status==5);
        if(status!=0)for(auto value:support)assert(value==0);
    }
    float a=0,b=0;
    assert(metal_ridge(&a,&a,1,1)==1);
    assert(run_geometry(&a,&b,1,1,0,0,.1f)==1);
    assert(run_operator(&a,&b,1,1,0,99,.1f)==1);
    a=std::numeric_limits<float>::quiet_NaN();
    assert(metal_ridge(&a,&b,1,1)==1);
    std::vector<double> strengths(300,0);std::vector<float> edges(300);
    for(size_t i=0;i<80;++i)strengths[i]=.15;
    strengths[0]=.8;strengths[99]=.15;
    geometry::hysteresis(strengths,edges.data(),100,3,.1,.5);
    assert(edges[79]==1 && edges[99]==0);
    double cloud[]={0,0,2, 0,0,1, 0,0,1, -1,0,1};
    double depth[4];int64_t index[4];
    assert(raster_depth(cloud,4,2,2,1,1,0,0,depth,index)==0);
    assert(depth[0]==1 && index[0]==1 && depth[1]==0 && index[1]==-1);
    cloud[0]=std::numeric_limits<double>::quiet_NaN();
    assert(raster_depth(cloud,4,2,2,1,1,0,0,depth,index)==1);
    assert(depth[0]==1 && index[0]==1);
    cloud[0]=0;
    assert(raster_depth(cloud,4,2,2,1,1,0,0,cloud,index)==1);
    double low[4],high[4],mean[4];int64_t count[4];
    assert(raster_elevation(cloud,4,-1,-1,1,1,1,2,2,count,low,high,mean)==0);
    assert(count[3]==3 && low[3]==1 && high[3]==2 && std::abs(mean[3]-4./3)<1e-14);
    assert(count[2]==1 && count[0]==0 && std::isnan(mean[0]));
    assert(raster_elevation(cloud,4,-1,-1,1,1,1,2,2,count,low,low,mean)==1);
    double organized[]={1,1,0,2,2,1,std::numeric_limits<double>::quiet_NaN(),0,2,2};
    int32_t labels[10]; double regions[16]; int32_t region_count=-1;
    assert(depth_connected_components(organized,5,2,1,0,3,.01,0,4,1,
                                      labels,regions,2,&region_count)==0);
    assert(region_count==2 && labels[0]==1 && labels[5]==1 && labels[2]==0 && labels[3]==2);
    assert(regions[0]==3 && regions[8]==4 && regions[1]==0 && regions[3]==1);
    labels[0]=123;
    assert(depth_connected_components(organized,5,2,1,0,3,-1,0,4,1,
                                      labels,regions,2,&region_count)==1);
    assert(labels[0]==123);
    int64_t voxel_keys[]={0,0,0, 1,0,0, 10,0,0};
    int32_t voxel_labels[3]={-7,-7,-7},voxel_count=-7;
    assert(voxel_connected_components(voxel_keys,3,6,1,voxel_labels,&voxel_count)==0);
    assert(voxel_count==2 && voxel_labels[0]==1 && voxel_labels[1]==1 && voxel_labels[2]==2);
    voxel_labels[0]=-7;
    assert(voxel_connected_components(voxel_keys,3,7,1,voxel_labels,&voxel_count)==1);
    assert(voxel_labels[0]==-7);
}
